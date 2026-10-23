// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The Track Table panel: every live point of the selected reconstruction as
//! one row, with the track table columns, sortable by a click on a heading
//! and filtered by a minimum and a maximum per column. The stored columns are
//! computed first and the computed ones a visible column, a bound or the sort
//! needs after them ([`job`]); a cell of a column not computed yet shows `…`.
//!
//! See `specs/gui/track-table.md`. The panel reads the track table cache
//! ([`crate::track_table`]) that `find_points` reads, and its quiet table job
//! ([`job`]) fills it. What it keeps between frames is its query, which is UI
//! state of the panel and not part of any version, and the **row list**: the
//! point indexes that pass the query, in its order. A reconstruction can hold
//! millions of points, so the row list is computed again only when its inputs
//! change (the query, the document serial or the columns the cache holds),
//! never once per frame, and only the visible rows are drawn
//! (`ScrollArea::show_rows`). The picture columns ([`pictures`]) are pinned
//! beside the Point column, set the row height, and make a picture only for a
//! row on screen. The *Crop* column's photographs are decoded, and the
//! references of points that store none picked ([`picks`]), only for the rows
//! on screen once the scroll position has stopped changing
//! ([`SCROLL_SETTLE_S`]).

pub(crate) mod job;
pub(crate) mod picks;
mod pictures;

#[cfg(test)]
mod tests;

use sfmtool_core::analysis::track_table::{
    query_points, ColumnBounds, ColumnCost, SortOrder, TrackTable, TrackTableColumn,
    TrackTableQueryOptions,
};
use std::cell::RefCell;
use std::sync::Arc;

use sfmtool_core::camera::image::ImageU8Pyramid;
use sfmtool_core::camera::PeekedPhotograph;
use sfmtool_core::EditedReconstruction;

use crate::document::VersionSerial;
use crate::patch_pictures::crop;
use crate::patch_pictures::marks::ProjectionOf;
use crate::scene::{PointRef, ReconId, SceneNode};
use crate::state::{AppState, DisplayPhotograph};
use crate::table_sort::{paint_sort_triangle, SortRule, TableSort, SORT_TRIANGLE_HALF_WIDTH};
use pictures::{
    hover_size, paint_picture, patch_image, Absent, CellPicture, Made, PictureColumn, PictureKey,
    PictureTextures, DEFAULT_PICTURE_SIZE, PICTURE_PAD, PICTURE_SIZES,
};

/// The wire names of the panel's picture columns, which `find_points`
/// refuses with a sentence of their own: they belong to the panel only.
#[cfg_attr(not(feature = "mcp"), allow(dead_code))]
pub(crate) fn picture_column_names() -> Vec<&'static str> {
    PictureColumn::ALL
        .iter()
        .map(|column| column.wire_name())
        .collect()
}

/// How many columns the track table has.
const COLUMN_COUNT: usize = TrackTableColumn::ALL.len();

/// The columns shown before any are hidden or shown: the point is always
/// shown, and of the stored columns the track's length, its error and its
/// distance, which is `∞` for a point at infinity, and of the computed ones
/// the ray angle. Repeat obs. is shown as well while the reconstruction has a
/// point with repeat observations ([`TrackTableQuery::is_shown`]).
const DEFAULT_COLUMNS: [TrackTableColumn; 4] = [
    TrackTableColumn::TrackLength,
    TrackTableColumn::RmsErrorPx,
    TrackTableColumn::CentroidDistance,
    TrackTableColumn::MaxRayAngleDeg,
];

/// The width of the Point column.
const POINT_WIDTH: f32 = 72.0;
/// The narrowest a value column is: room for its two bound boxes.
const MIN_VALUE_WIDTH: f32 = 84.0;
/// Space between a cell's text and its column's edges.
const CELL_PAD: f32 = 6.0;

/// How long the rows on screen must stay the same before the *Crop* column
/// asks for their photographs and queues reference picks for them, in
/// seconds. While the rows are still changing, as a drag of the scroll bar or
/// a turn of the wheel moves them on every frame, the cells draw the crops
/// already held and ask for nothing, so scrolling through a large table
/// sorted by a number does not queue a decode for every photograph it passes.
/// See `specs/gui/track-table.md` § "The Crop column" for the measurement it
/// was chosen from.
pub(crate) const SCROLL_SETTLE_S: f64 = 0.15;

impl SortRule for TrackTableColumn {
    /// The columns whose large values are the bad end: the repeat
    /// observations, the error, the condition number and the self-similarity
    /// radius.
    fn worst_first_descending(self) -> bool {
        matches!(
            self,
            TrackTableColumn::RepeatObservations
                | TrackTableColumn::RmsErrorPx
                | TrackTableColumn::ConditionNumber
                | TrackTableColumn::LargestZnccSelfSimilarityRadius
        )
    }
}

/// Whether a column's bad end is its small values, so that its first click,
/// which orders it increasing, puts the worst rows first. The inverse-depth z
/// is among them: it is the depth over its uncertainty, and a small one is a
/// point that cannot be told from one at infinity.
fn worst_first_ascending(column: TrackTableColumn) -> bool {
    matches!(
        column,
        TrackTableColumn::TrackLength
            | TrackTableColumn::MaxRayAngleDeg
            | TrackTableColumn::InverseDepthZ
            | TrackTableColumn::LowestZncc
            | TrackTableColumn::NormalConfidence
    )
}

/// The heading a column is shown under.
fn column_title(column: TrackTableColumn) -> &'static str {
    match column {
        TrackTableColumn::TrackLength => "Track",
        TrackTableColumn::RepeatObservations => "Repeat obs.",
        TrackTableColumn::RmsErrorPx => "Error (px)",
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
        TrackTableColumn::RepeatObservations => {
            "The track's observations beyond one per image: Track less the number of distinct \
             images that see the point. Anything but 0 is a defect, one image holding several \
             observations of the point, which a GLOMAP solve can produce. The column is shown \
             while the reconstruction has such a point, unless it was unticked here."
        }
        TrackTableColumn::RmsErrorPx => {
            "The point's stored RMS reprojection error over its observations, in pixels."
        }
        TrackTableColumn::X | TrackTableColumn::Y | TrackTableColumn::Z => {
            "The point's position along this axis, in the reconstruction's unit. A point at \
             infinity has no value."
        }
        TrackTableColumn::CentroidDistance => {
            "The point's distance from the centroid of the posed cameras, in the unit of X. A \
             point at infinity, a direction rather than a position, is at \u{221e}: it sorts \
             above every finite distance, a finite maximum drops it, and a minimum of inf \
             keeps only the points at infinity."
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
            "The largest angle between two of the track's viewing rays, the rays from each \
             observing camera's centre to the point, in degrees. A small angle leaves the \
             point's depth poorly fixed. A point at infinity is at 0: every sighting of a \
             direction casts the same ray. Computed after the stored columns."
        }
        TrackTableColumn::ConditionNumber => {
            "The condition number of the point's triangulation from its viewing rays. A large \
             one means the rays are close to parallel and the position is poorly fixed. A \
             point at infinity, or a track with fewer than two usable rays, has no value. \
             Computed after the stored columns."
        }
        TrackTableColumn::InverseDepthZ => {
            "The point's depth over its one-sigma depth uncertainty, from the same \
             triangulation, with each ray's noise its error (at least 1 px) over the focal \
             length. Below about 3 to 4 the point cannot be told from one at infinity. A point \
             at infinity, or a track with fewer than two usable rays, has no value. Computed \
             after the stored columns."
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

/// Whether the panel offers `column` for `edited`: a column the
/// reconstruction can supply. Every computed column is offered for every
/// reconstruction.
fn offered(column: TrackTableColumn, edited: &EditedReconstruction) -> bool {
    column.is_available(edited)
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
    /// The picture columns shown, in [`PictureColumn::ALL`] order. Every one
    /// is shown by default, for a reconstruction that can supply it.
    pictures: Vec<PictureColumn>,
    /// Whether only the points the selected image observes are kept.
    seen_in_selected_image: bool,
    /// Whether Repeat obs. has been ticked or unticked in the Columns menu.
    /// Until it has, the column is shown exactly while the reconstruction
    /// on show has a point with repeat observations; after, `shown` alone
    /// decides, so the panel never undoes the person's choice.
    repeat_observations_chosen: bool,
}

impl Default for TrackTableQuery {
    fn default() -> Self {
        Self {
            bounds: std::array::from_fn(|_| BoundText::default()),
            sort: None,
            shown: DEFAULT_COLUMNS.to_vec(),
            pictures: PictureColumn::ALL.to_vec(),
            seen_in_selected_image: false,
            repeat_observations_chosen: false,
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

    /// Whether `column` is drawn, where `repeats` says whether the
    /// reconstruction on show has a point with repeat observations. A column
    /// in `shown` is drawn. Repeat obs. is drawn as well, until the person
    /// ticks or unticks it in the Columns menu, while `repeats` is true or
    /// while it holds a bound or the sort, so that it is not hidden while it
    /// filters or orders the rows.
    fn is_shown(&self, column: TrackTableColumn, repeats: bool) -> bool {
        self.shown.contains(&column)
            || (column == TrackTableColumn::RepeatObservations
                && !self.repeat_observations_chosen
                && (repeats || self.holds(column)))
    }

    /// The computed columns the table job has to compute for this query: the
    /// ones shown, and the ones holding a bound or the sort (which are shown
    /// too), in [`TrackTableColumn::ALL`] order. A computed column none of
    /// those needs is never computed for the panel.
    fn needed_computed(&self) -> Vec<TrackTableColumn> {
        TrackTableColumn::ALL
            .into_iter()
            .filter(|&c| c.cost() == ColumnCost::Computed)
            .filter(|&c| self.shown.contains(&c) || self.holds(c))
            .collect()
    }

    /// The columns drawn, in [`TrackTableColumn::ALL`] order
    /// ([`Self::is_shown`]).
    fn visible(&self, repeats: bool) -> Vec<TrackTableColumn> {
        TrackTableColumn::ALL
            .into_iter()
            .filter(|&c| self.is_shown(c, repeats))
            .collect()
    }

    /// Tick or untick `column` in the Columns menu: [`Self::set_shown`], and
    /// for Repeat obs. the end of showing it by itself.
    fn choose_shown(&mut self, column: TrackTableColumn, shown: bool, repeats: bool) {
        if column == TrackTableColumn::RepeatObservations && !self.repeat_observations_chosen {
            if self.is_shown(column, repeats) && !shown && self.holds(column) {
                return;
            }
            self.repeat_observations_chosen = true;
        }
        self.set_shown(column, shown);
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

    /// Show or hide the picture column `column`.
    fn set_picture_shown(&mut self, column: PictureColumn, shown: bool) {
        self.pictures.retain(|&c| c != column);
        if shown {
            self.pictures.push(column);
            self.pictures.sort();
        }
    }

    /// The picture columns drawn for `edited`: the shown ones it can supply.
    fn visible_pictures(&self, edited: &EditedReconstruction) -> Vec<PictureColumn> {
        self.pictures
            .iter()
            .copied()
            .filter(|c| c.is_available(edited))
            .collect()
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
    /// How many live points of the table the rows were computed from have
    /// repeat observations, `0` while the table does not hold the column.
    repeats: usize,
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
        // Counted with the rows, so once per change of the table rather than
        // once per frame.
        self.repeats = table
            .values(TrackTableColumn::RepeatObservations)
            .map_or(0, |values| values.iter().filter(|&&v| v > 0.0).count());
        self.inputs = Some(inputs);
        self.recomputed += 1;
        moved
    }
}

/// The label of the box that keeps only the points the selected image
/// observes. It does not name the image, so a long image name does not push
/// the header onto a second line; the hover text names it.
const SEEN_IN_SELECTED_IMAGE: &str = "Seen in selected image";

/// The hover text of the [`SEEN_IN_SELECTED_IMAGE`] box: what it keeps, and
/// the selected image's name, or that no image of the reconstruction is
/// selected and the box filters nothing.
fn seen_in_selected_image_hover(image_name: Option<&str>) -> String {
    match image_name {
        Some(name) => format!("Keep only the points seen in the selected image, {name}."),
        None => "Keep only the points seen in the selected image. No image of this \
                 reconstruction is selected, so this filters nothing."
            .to_string(),
    }
}

/// The name of the picture size's dropdown, which is also the label drawn
/// before it.
const SIZE_NAME: &str = "Size";

/// A picture size as the dropdown writes it: `64 pt`.
fn size_text(size: f32) -> String {
    format!("{size} pt")
}

/// The *Size* label and the dropdown of [`PICTURE_SIZES`] over `size`.
///
/// The dropdown carries [`SIZE_NAME`] as its AccessKit label, which is the
/// name `get_widgets` lists it under, and the size it shows as its value.
/// egui's `labelled_by`, which links it to the label for a screen reader,
/// leaves the label empty.
fn size_dropdown(ui: &mut egui::Ui, size: &mut f32) {
    let label = ui.label(SIZE_NAME);
    // As wide as the widest size and the arrow, not egui's default combo
    // width, so the header keeps its room; the same for every size, so the
    // header does not shift when the size changes.
    let widest = PICTURE_SIZES
        .iter()
        .map(|&choice| {
            egui::WidgetText::from(size_text(choice))
                .into_galley(
                    ui,
                    Some(egui::TextWrapMode::Extend),
                    f32::INFINITY,
                    egui::TextStyle::Button,
                )
                .size()
                .x
        })
        .fold(0.0, f32::max);
    let spacing = ui.spacing();
    let width = widest + spacing.icon_spacing + spacing.icon_width + 2.0 * spacing.button_padding.x;
    let response = egui::ComboBox::from_id_salt("track_table_picture_size")
        .width(width)
        .selected_text(size_text(*size))
        .show_ui(ui, |ui| {
            for choice in PICTURE_SIZES {
                ui.selectable_value(size, choice, size_text(choice));
            }
        })
        .response
        .labelled_by(label.id);
    ui.ctx()
        .accesskit_node_builder(response.id, |node| node.set_label(SIZE_NAME));
    response
        .on_hover_text(
            "The size of the pictures, which sets the height of the rows. The rows at the top \
             stay there when it changes.",
        )
        .on_disabled_hover_text(
            "The size of the pictures. No picture column is shown, so the rows are a line of \
             text high.",
        );
}

/// What the query cannot apply to the reconstruction on show, said in the
/// header so that nothing filters unseen and nothing is dropped unsaid.
#[derive(Debug, Default, PartialEq)]
struct Unapplied {
    /// Columns holding a bound or the sort that the reconstruction cannot
    /// supply: kept, and applied again once one that can is selected.
    unsupplied: Vec<TrackTableColumn>,
    /// Columns holding a bound that the cache does not hold yet.
    pending_bounds: Vec<TrackTableColumn>,
    /// The sort's column, when the cache does not hold it yet.
    pending_sort: Option<TrackTableColumn>,
}

impl Unapplied {
    /// Whether a bound or the sort waits for a column the cache does not hold
    /// yet.
    fn is_pending(&self) -> bool {
        !self.pending_bounds.is_empty() || self.pending_sort.is_some()
    }

    /// The columns a bound or the sort waits for, in
    /// [`TrackTableColumn::ALL`] order.
    fn pending(&self) -> Vec<TrackTableColumn> {
        TrackTableColumn::ALL
            .into_iter()
            .filter(|c| self.pending_bounds.contains(c) || self.pending_sort == Some(*c))
            .collect()
    }
}

/// The picture size the *Size* dropdown sets, in points: the side of each
/// picture, and with [`PICTURE_PAD`] above and below, the row height while a
/// picture column is shown. Panel state like the query, and not saved.
#[derive(Debug, Clone, Copy, PartialEq)]
struct PictureSize(f32);

impl Default for PictureSize {
    fn default() -> Self {
        Self(DEFAULT_PICTURE_SIZE)
    }
}

/// Whether the rows on screen have stopped changing: the range of rows the
/// last frame drew and when it last changed, in egui's time.
#[derive(Debug, Clone, Default)]
struct ScrollMotion {
    /// The rows the last frame drew, by their place in the row list.
    range: Option<std::ops::Range<usize>>,
    /// When `range` last changed, or `None` while it has not changed since
    /// the panel first drew rows.
    changed_at: Option<f64>,
}

impl ScrollMotion {
    /// Note that the frame at `now` draws `range`, and answer how long until
    /// the rows count as stopped: `None` once they have stayed the same for
    /// [`SCROLL_SETTLE_S`], or from the first frame that draws rows.
    fn observe(&mut self, range: std::ops::Range<usize>, now: f64) -> Option<f64> {
        if self.range.as_ref().is_some_and(|last| *last != range) {
            self.changed_at = Some(now);
        }
        self.range = Some(range);
        let waited = now - self.changed_at?;
        (waited < SCROLL_SETTLE_S).then_some(SCROLL_SETTLE_S - waited)
    }
}

/// The Track Table panel's state: its query, the row list, the picture size
/// and the pictures drawn, and what it remembers of the last frame to keep the
/// selection, the hover and the scroll position in step.
#[derive(Debug, Default)]
pub(crate) struct TrackTablePanel {
    /// The query, which is panel state.
    query: TrackTableQuery,
    /// The rows that pass it, kept between frames.
    rows: RowList,
    /// The picture size the header's *Size* dropdown sets.
    picture_size: PictureSize,
    /// The textures of the pictures drawn, kept between frames.
    pictures: PictureTextures,
    /// The row height and the vertical scroll offset of the last frame's
    /// rows, so that a new row height (the *Size* dropdown, or a picture column
    /// shown or hidden) keeps the same rows at the top.
    scroll: Option<(f32, f32)>,
    /// Whether the rows on screen have stopped changing, which the *Crop*
    /// column waits for before it asks for photographs and picks.
    motion: ScrollMotion,
    /// The selected point as of the last frame, so that a selection made
    /// elsewhere scrolls its row into view once.
    seen_selection: Option<PointRef>,
    /// Whether the pointer was over the rows in the last frame, so that
    /// leaving them clears the hover once rather than every frame.
    had_pointer: bool,
    /// The rows the last frame drew, each with its rectangle.
    #[cfg(test)]
    drawn: Vec<(u32, egui::Rect)>,
    /// The picture cells the last frame drew: the row's point, the column
    /// and what the cell drew.
    #[cfg(test)]
    drawn_pictures: Vec<(u32, PictureColumn, CellPicture)>,
    /// Every image whose photograph a *Crop* cell asked to have decoded, in
    /// order, over every frame.
    #[cfg(test)]
    requested_photographs: Vec<usize>,
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
    /// The picture cells drawn, for a test to read.
    #[cfg(test)]
    pictures: Vec<(u32, PictureColumn, CellPicture)>,
    /// The points of the drawn rows whose crops wait for a reference pick,
    /// with their node and version, on a frame whose rows have stopped
    /// changing and are of the version at the cursor; `None` otherwise.
    picks: Option<(ReconId, VersionSerial, Vec<u32>)>,
    /// The images whose photographs the drawn crops asked to have decoded.
    #[cfg(test)]
    requested: Vec<usize>,
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
        let mut bounds = Vec::new();
        for column in TrackTableColumn::ALL {
            let text = self.query.bound(column);
            if !text.is_set() {
                continue;
            }
            if !offered(column, edited) {
                unapplied.unsupplied.push(column);
            } else if !table.has(column) {
                unapplied.pending_bounds.push(column);
            } else {
                bounds.extend(text.parsed(column));
            }
        }
        let order_by = self.query.sort.and_then(|sort| {
            let usable = if !offered(sort.column, edited) {
                if !unapplied.unsupplied.contains(&sort.column) {
                    unapplied.unsupplied.push(sort.column);
                }
                false
            } else if !table.has(sort.column) {
                unapplied.pending_sort = Some(sort.column);
                false
            } else {
                true
            };
            usable.then_some((
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
        let before = self.query.clone();
        #[cfg_attr(not(test), allow(unused_mut))]
        let mut asked = self.show_node(ui, state, id);
        // After the frame's edits of the query, so a column ticked or a
        // heading clicked in this frame is computed from the drive that
        // follows the egui pass.
        state.want_track_table_columns(id, self.query.needed_computed());
        // The rows are computed from the query at the top of a frame, so an
        // edit this frame made shows in the next one, which is asked for here
        // rather than left to the next input event.
        if self.query != before {
            ui.ctx().request_repaint();
        }
        #[cfg(test)]
        {
            self.drawn = std::mem::take(&mut asked.drawn);
            self.drawn_pictures = std::mem::take(&mut asked.pictures);
            self.requested_photographs.append(&mut asked.requested);
        }
        if let Some((node, document, points)) = asked.picks.take() {
            state.want_reference_picks(node, document, points);
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
        let current = state.track_table(id);
        // While the table of a new version is computed, the table the row
        // list was computed from is kept on show, marked out of date: the
        // cache holds it until the new one replaces it.
        let previous = match current {
            Some(_) => None,
            None => state.track_tables.latest(id).and_then(|(document, table)| {
                let inputs = self.rows.inputs.as_ref()?;
                (inputs.node == id && inputs.document == document && self.rows.error.is_none())
                    .then_some(table)
            }),
        };
        let stale = previous.is_some();
        let table = current.or(previous);

        // The rows first, so the header counts the rows this frame draws. Out
        // of date, they are not computed again: the old table's rows and the
        // new version's observations do not go together, so a change of the
        // query waits for the new table. While a bound or the sort waits for a
        // column the cache does not hold yet, rows already computed for this
        // version stay as they are rather than being ordered or filtered by
        // part of the query; with none, as when the table first lands, the
        // rest of the query is applied.
        let mut reordered = false;
        let unapplied = table.map(|table| {
            let (inputs, unapplied) = self.row_inputs(node, table, selected_image);
            let waiting = unapplied.is_pending()
                && self.rows.error.is_none()
                && self
                    .rows
                    .inputs
                    .as_ref()
                    .is_some_and(|old| old.node == id && old.document == inputs.document);
            if !stale && !waiting {
                reordered = self.rows.refresh(edited, table, inputs);
            }
            unapplied
        });

        self.show_header(ui, node, table, stale, selected_image);
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

        let layout = ColumnLayout::new(
            ui,
            &self.query.visible_pictures(edited),
            &self.query.visible(self.rows.repeats > 0),
            edited,
            self.picture_size.0,
        );
        let selected = state.selected_point.filter(|p| p.recon == id);
        // A selection made elsewhere scrolls its row into view; otherwise a
        // new sort, new bounds or another reconstruction starts at the top,
        // where the rows the query puts first are. Out of date, the rows may
        // name points the new version deleted, so the selection is looked for
        // once the new table lands.
        let scroll_to = if stale {
            None
        } else if state.selected_point != self.seen_selection {
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
        // the columns under them, as Track View's headings do. The Point
        // column is pinned at the panel's left edge, in the rows and above
        // them.
        let heading_band = reserve_band(ui, heading_height(ui));
        let bounds_band = reserve_band(ui, bounds_height(ui));
        let mut left = heading_band.left();
        if self.rows.rows.is_empty() {
            ui.label(egui::RichText::new("No point passes the bounds.").weak());
        } else {
            let document = self
                .rows
                .inputs
                .as_ref()
                .expect("rows on show were computed")
                .document;
            let rows = RowsView {
                state,
                node,
                table,
                rows: &self.rows.rows,
                document,
                layout: &layout,
                selected,
                hovered,
                stale,
                wanted_picks: RefCell::new(Vec::new()),
                requested: RefCell::new(Vec::new()),
            };
            self.pictures.begin_frame();
            let shown = rows.show(
                ui,
                scroll_to,
                self.scroll,
                &mut self.motion,
                &mut self.pictures,
                &mut asked,
            );
            self.pictures.end_frame();
            if shown.settled && !stale {
                asked.picks = Some((id, document, rows.wanted_picks.take()));
            }
            #[cfg(test)]
            asked.requested.append(&mut rows.requested.borrow_mut());
            self.scroll = Some(shown.scroll);
            left = shown.left;
            let over = shown.over;
            if over {
                self.had_pointer = true;
            } else if std::mem::take(&mut self.had_pointer) {
                asked.hover = Some(None);
            }
        }
        // Out of date, the headings and the boxes stay live: an edit of them
        // changes the query, and the rows follow it once the new table lands.
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
        stale: bool,
        selected_image: Option<u32>,
    ) {
        let edited = node.edited();
        // The count of the table on show, which the row list was computed
        // from.
        let repeats = if table.is_some() {
            self.rows.repeats
        } else {
            0
        };
        ui.horizontal_wrapped(|ui| {
            ui.label(egui::RichText::new(&node.label).strong());
            if let Some(table) = table {
                ui.label(format!(
                    "{} of {} points",
                    grouped(self.rows.rows.len()),
                    grouped(table.point_indexes().len())
                ));
            }
            if repeats > 0 {
                ui.label(format!(
                    "\u{b7} {} with repeat observations",
                    grouped(repeats)
                ))
                .on_hover_text(
                    "Live points whose track holds more than one observation in one image, \
                     whatever the bounds. That is a defect in the reconstruction, which a \
                     GLOMAP solve can produce. Click the Repeat obs. heading to put them first.",
                );
            }
            if stale {
                ui.spinner();
                ui.label(egui::RichText::new("Updating\u{2026}").weak())
                    .on_hover_text(
                        "The reconstruction has changed. These rows and the count are of the \
                         version before the change, and they are replaced when the track table \
                         of the version now shown is ready. Until then a row cannot be selected \
                         or hovered, since it may name a point that no longer exists, and a \
                         change of the bounds or the sort applies once the new rows arrive.",
                    );
            }
            let query = &mut self.query;
            egui::containers::menu::MenuButton::new("Columns")
                .config(
                    egui::containers::menu::MenuConfig::new()
                        .close_behavior(egui::PopupCloseBehavior::CloseOnClickOutside),
                )
                .ui(ui, |ui| {
                    // The picture columns first, as they are drawn.
                    for column in PictureColumn::ALL {
                        if !column.is_available(edited) {
                            continue;
                        }
                        let mut shown = query.pictures.contains(&column);
                        let response = ui
                            .checkbox(&mut shown, column.title())
                            .on_hover_text(column.description());
                        if response.changed() {
                            query.set_picture_shown(column, shown);
                        }
                    }
                    for column in TrackTableColumn::ALL {
                        if !offered(column, edited) {
                            continue;
                        }
                        let mut shown = query.is_shown(column, repeats > 0);
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
                            query.choose_shown(column, shown, repeats > 0);
                        }
                    }
                });
            // The picture size, for a reconstruction with a picture column;
            // greyed while none is shown, since the rows are then text.
            if PictureColumn::ALL.iter().any(|c| c.is_available(edited)) {
                let pictures_shown = !query.visible_pictures(edited).is_empty();
                ui.add_enabled_ui(pictures_shown, |ui| {
                    // One group, so that a wrapping header line keeps the
                    // label beside its dropdown.
                    ui.horizontal(|ui| size_dropdown(ui, &mut self.picture_size.0));
                });
            }
            let image_name = selected_image.and_then(|index| {
                node.recon()
                    .image_set
                    .images
                    .get(index as usize)
                    .map(|image| image.name.clone())
            });
            // A fixed label, so the line does not grow with the image's name;
            // the hover text names the image.
            let mut text = egui::RichText::new(SEEN_IN_SELECTED_IMAGE);
            if image_name.is_none() && self.query.seen_in_selected_image {
                text = text.weak();
            }
            ui.checkbox(&mut self.query.seen_in_selected_image, text)
                .on_hover_text(seen_in_selected_image_hover(image_name.as_deref()));
        });
    }

    /// The heading row: a click on a value column's heading orders the rows
    /// by it, and a click on *Point* puts them back in point order. A picture
    /// column's heading takes no click, since its pictures have no order.
    fn show_headings(
        &mut self,
        ui: &mut egui::Ui,
        band: egui::Rect,
        left: f32,
        layout: &ColumnLayout,
    ) {
        let pinned_ui = &mut band_ui(ui, band);
        let scrolled_ui = &mut band_ui(ui, scrolled_part(band, layout.pinned_width));
        let height = band.height();
        let font = egui::TextStyle::Body.resolve(pinned_ui.style());
        for &(slot, x, width) in &layout.slots {
            let (ui, cell_left) = match slot {
                Slot::Point | Slot::Picture(_) => (&mut *pinned_ui, band.left() + x),
                Slot::Value(_) => (&mut *scrolled_ui, left + x),
            };
            let cell = egui::Rect::from_min_size(
                egui::pos2(cell_left, band.min.y),
                egui::vec2(width, height),
            );
            if let Slot::Picture(picture) = slot {
                ui.painter().text(
                    egui::pos2(cell.min.x + CELL_PAD, cell.center().y),
                    egui::Align2::LEFT_CENTER,
                    picture.title(),
                    font.clone(),
                    ui.visuals().text_color(),
                );
                let response = ui.interact(
                    cell,
                    ui.id().with(("track_table_heading", slot)),
                    egui::Sense::hover(),
                );
                response.widget_info(|| {
                    egui::WidgetInfo::labeled(egui::WidgetType::Label, true, picture.title())
                });
                response.on_hover_text(picture.description());
                continue;
            }
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
                Slot::Value(column) => (column_title(column), Some(column)),
                _ => ("Point", None),
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
        // The pinned columns have no boxes, and the value columns' boxes pass
        // under them, as their cells do.
        let ui = &mut band_ui(ui, scrolled_part(band, layout.pinned_width));
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

/// The part of `band` right of the pinned columns, `pinned_width` wide,
/// where the value columns scroll sideways.
fn scrolled_part(band: egui::Rect, pinned_width: f32) -> egui::Rect {
    let mut part = band;
    part.min.x = (band.min.x + pinned_width).min(band.max.x);
    part
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
    if unapplied.is_pending() {
        let pending = unapplied.pending();
        ui.label(egui::RichText::new(waiting_text(unapplied)).weak())
            .on_hover_text(format!(
                "{} {} still being computed. Until then the rows are not filtered or sorted by \
                 {}: rows already listed stay as they are, and a change of the bounds or the \
                 sort applies once {} ready.",
                titles(&pending),
                if pending.len() == 1 { "is" } else { "are" },
                if pending.len() == 1 { "it" } else { "them" },
                if pending.len() == 1 {
                    "it is"
                } else {
                    "they are"
                },
            ));
    }
}

/// The line that says the query waits for columns the cache does not hold
/// yet: *Sorting when Ray angle (°) is ready.*, *Filtering when … is ready.*
/// or *Filtering and sorting when … are ready.*, by what waits.
fn waiting_text(unapplied: &Unapplied) -> String {
    let pending = unapplied.pending();
    let doing = match (
        !unapplied.pending_bounds.is_empty(),
        unapplied.pending_sort.is_some(),
    ) {
        (true, true) => "Filtering and sorting",
        (true, false) => "Filtering",
        _ => "Sorting",
    };
    format!(
        "{doing} when {} {} ready.",
        pending
            .iter()
            .map(|&c| column_title(c))
            .collect::<Vec<_>>()
            .join(", "),
        if pending.len() == 1 { "is" } else { "are" }
    )
}

/// One column the table draws.
///
/// The Point column and the value columns are text; a picture column draws a
/// picture per row, and while one is shown the row height follows the
/// picture size rather than the text.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Slot {
    /// The point index, the row's key.
    Point,
    /// A picture column, pinned beside the Point column.
    Picture(PictureColumn),
    /// One track table column.
    Value(TrackTableColumn),
}

/// The columns drawn, left to right, each with its left edge and its width
/// from the table's left edge, and the height of every row.
struct ColumnLayout {
    slots: Vec<(Slot, f32, f32)>,
    width: f32,
    /// The width of the pinned columns: the Point column and the picture
    /// columns, which stay at the panel's left edge while the value columns
    /// scroll sideways under them.
    pinned_width: f32,
    /// The side of each picture.
    picture_size: f32,
    /// Every row's height: the picture size and [`PICTURE_PAD`] above and
    /// below while a picture column is shown, or a line of text where that is
    /// taller or while none is shown.
    row_height: f32,
}

impl ColumnLayout {
    /// The Point column, then each picture column of `pictures` at
    /// `picture_size`, then each shown column `edited` can supply, each value
    /// column wide enough for its heading, its sort triangle and its two bound
    /// boxes.
    fn new(
        ui: &egui::Ui,
        pictures: &[PictureColumn],
        shown: &[TrackTableColumn],
        edited: &EditedReconstruction,
        picture_size: f32,
    ) -> Self {
        let font = egui::TextStyle::Body.resolve(ui.style());
        let title_width = |title: &str| {
            ui.painter()
                .layout_no_wrap(title.to_string(), font.clone(), egui::Color32::WHITE)
                .size()
                .x
        };
        let mut slots = vec![(Slot::Point, 0.0, POINT_WIDTH)];
        let mut x = POINT_WIDTH;
        for &picture in pictures {
            let width = (picture_size.max(title_width(picture.title())) + 2.0 * CELL_PAD).round();
            slots.push((Slot::Picture(picture), x, width));
            x += width;
        }
        let pinned_width = x;
        for &column in shown {
            if !offered(column, edited) {
                continue;
            }
            let title = title_width(column_title(column));
            let width =
                (title + 2.0 * CELL_PAD + 4.0 * SORT_TRIANGLE_HALF_WIDTH).max(MIN_VALUE_WIDTH);
            slots.push((Slot::Value(column), x, width));
            x += width;
        }
        let row_height = if pictures.is_empty() {
            text_row_height(ui)
        } else {
            (picture_size + 2.0 * PICTURE_PAD).max(text_row_height(ui))
        };
        Self {
            slots,
            width: x,
            pinned_width,
            picture_size,
            row_height,
        }
    }
}

/// The row height while no picture column is shown: a line of text.
fn text_row_height(ui: &egui::Ui) -> f32 {
    ui.text_style_height(&egui::TextStyle::Body) + 6.0
}

/// The rows, as one frame draws them.
struct RowsView<'a> {
    /// The state, for the photograph cache and the reference picks the
    /// *Crop* cells read.
    state: &'a AppState,
    node: &'a SceneNode,
    table: &'a TrackTable,
    rows: &'a [u32],
    /// The document serial of the version the rows belong to, which names
    /// their pictures: the node's serial at its cursor while the rows are up
    /// to date, the version before an edit while they are not.
    document: VersionSerial,
    layout: &'a ColumnLayout,
    selected: Option<PointRef>,
    hovered: Option<PointRef>,
    /// Whether the rows are of a version the node has left, kept on show
    /// while the new version's table is computed. Such rows are dimmed and
    /// take no click and no hover: one may name a point the new version
    /// deleted. Their pictures are the ones already drawn for that version,
    /// dimmed; none is made for them.
    stale: bool,
    /// The points whose crops wait for a reference pick, gathered as the
    /// cells are drawn on a frame whose rows have stopped changing.
    wanted_picks: RefCell<Vec<u32>>,
    /// The images whose photographs the crops drawn asked to have decoded.
    requested: RefCell<Vec<usize>>,
}

/// What drawing the rows answers.
struct RowsShown {
    /// Whether the pointer is over the rows.
    over: bool,
    /// The left edge the rows were drawn from, which the headings follow.
    left: f32,
    /// The row height and the vertical scroll offset the rows were drawn at.
    scroll: (f32, f32),
    /// Whether the rows drawn have stopped changing, so the crops asked for
    /// their photographs and picks.
    settled: bool,
}

/// One picture cell a row drew: its column, its rectangle and its picture.
type DrawnPicture = (PictureColumn, egui::Rect, CellPicture);

impl RowsView<'_> {
    /// Draw the visible rows, scrolling row `scroll_to` into view. With no
    /// row to scroll to and a row height other than the last frame's
    /// (`last_scroll`, the height and offset it was drawn at), the offset is
    /// scaled to the new height, so the same rows stay at the top. A click
    /// and a hover go into `asked`. The pinned cells are drawn at the
    /// viewport's left edge rather than the rows', so they stay there while
    /// the rest scrolls sideways. The pictures of the drawn rows come from
    /// `pictures`, which makes the ones it lacks for rows that are up to
    /// date. `motion` says whether the rows drawn have stopped changing; until
    /// they have, a crop asks for no photograph and no pick, and a repaint is
    /// asked for the moment they will count as stopped.
    fn show(
        &self,
        ui: &mut egui::Ui,
        scroll_to: Option<usize>,
        last_scroll: Option<(f32, f32)>,
        motion: &mut ScrollMotion,
        pictures: &mut PictureTextures,
        asked: &mut Asked,
    ) -> RowsShown {
        let row_height = self.layout.row_height;
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
        } else if let Some((last_height, last_offset)) =
            last_scroll.filter(|&(height, _)| height != row_height && height > 0.0)
        {
            area = area.vertical_scroll_offset(last_offset / last_height * row_height);
        }
        let pinned_left = ui.available_rect_before_wrap().left();
        let mut drawn_left = pinned_left;
        let now = ui.input(|i| i.time);
        let mut settled = true;
        let output = area.show_rows(ui, row_height, self.rows.len(), |ui, range| {
            ui.set_min_width(self.layout.width);
            let top = ui.max_rect().top();
            let left = ui.max_rect().left();
            drawn_left = left;
            if let Some(wait) = motion.observe(range.clone(), now) {
                settled = false;
                ui.ctx()
                    .request_repaint_after(std::time::Duration::from_secs_f64(wait));
            }
            let mut hovered_row = None;
            for (k, position) in range.clone().enumerate() {
                let point = self.rows[position];
                let rect = egui::Rect::from_min_size(
                    egui::pos2(left, top + k as f32 * row_height),
                    egui::vec2(self.layout.width, row_height),
                );
                // The whole row, the pinned cells with it: they lie inside the
                // row's rectangle wherever it is scrolled to.
                let sense = if self.stale {
                    egui::Sense::hover()
                } else {
                    egui::Sense::click()
                };
                let response = ui.interact(rect, ui.id().with(("track_table_row", point)), sense);
                let reference = PointRef {
                    recon: self.node.id,
                    point,
                };
                let selected = self.selected == Some(reference);
                let cells = self.paint_row(
                    ui,
                    rect,
                    pinned_left,
                    position,
                    reference,
                    pictures,
                    settled,
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
                {
                    asked.drawn.push((point, rect));
                    asked.pictures.extend(
                        cells
                            .iter()
                            .map(|&(column, _, picture)| (point, column, picture)),
                    );
                }
                if response.clicked() {
                    asked.select = Some(reference);
                }
                if response.hovered() && !self.stale {
                    hovered_row = Some(reference);
                    // Minted for the row under the pointer only: an id walks
                    // the node's version graph.
                    let id = crate::scene::point_id(self.node, point as usize);
                    let under = ui
                        .ctx()
                        .pointer_hover_pos()
                        .and_then(|at| cells.iter().find(|(_, cell, _)| cell.contains(at)));
                    match under {
                        Some(&(column, _, picture)) => {
                            let size = hover_size(self.layout.picture_size);
                            let key = self.key(column, point);
                            response.on_hover_ui_at_pointer(|ui| {
                                self.show_picture_hover(ui, key, picture, size, pictures, &id);
                            });
                        }
                        None => {
                            response.on_hover_text_at_pointer(format!("{id}. Click to select it."));
                        }
                    }
                }
            }
            hovered_row
        });
        // The hover position rather than `rect_contains_pointer`, which reads
        // the last position a pointer that has left the window had; and only
        // where no popup or window lies over the rows.
        // Out of date, the rows do not own the hover: a pointer over them
        // counts as over none, so leaving the live rows clears it once.
        let over = !self.stale
            && ui.ctx().pointer_hover_pos().is_some_and(|at| {
                output.inner_rect.contains(at) && ui.ctx().layer_id_at(at) == Some(ui.layer_id())
            });
        if over {
            asked.hover = Some(output.inner);
        }
        RowsShown {
            over,
            left: drawn_left,
            scroll: (row_height, output.state.offset.y),
            settled,
        }
    }

    /// The cache key of `point`'s cell in `column`, for the version the rows
    /// belong to.
    fn key(&self, column: PictureColumn, point: u32) -> PictureKey {
        PictureKey {
            column,
            node: self.node.id,
            point,
            document: self.document,
        }
    }

    /// The *Crop* picture of `point`: its reference photograph cut around its
    /// patch's outline, or why there is none, or [`Made::NotYet`] while the
    /// photograph is decoding or the reference is being picked.
    ///
    /// The photograph is the point's reference observation's. A point that
    /// stores none gets the reference-view rule's pick ([`picks`]), unless a
    /// bitmap writer already ran the rule for it and found no eligible view
    /// ([`rule_found_no_reference`]): then the cell says *no reference* at
    /// once. Only while `settled` does a crop ask for a pick or for its
    /// photograph to be decoded; until then it draws what is held.
    fn crop_picture(&self, ctx: &egui::Context, point: u32, settled: bool) -> Made {
        let edited = self.node.edited();
        let recon = self.node.recon();
        let Some(view) = edited.point(point) else {
            return Made::Absent(Absent::NoPatch);
        };
        let Some(patch_frame) = view.patch_frame() else {
            return Made::Absent(Absent::NoPatch);
        };
        let stored = view
            .reference_observation()
            .and_then(|r| u32::try_from(r).ok());
        let reference = match stored {
            Some(reference) => reference,
            // A bitmap writer ran the rule for this point and found nothing.
            None if rule_found_no_reference(&view) => return Made::Absent(Absent::NoReference),
            None => match self
                .state
                .reference_pick(self.node.id, point, self.document)
            {
                Some(Some(reference)) => reference,
                Some(None) => return Made::Absent(Absent::NoReference),
                None => {
                    if settled {
                        self.wanted_picks.borrow_mut().push(point);
                    }
                    return Made::NotYet;
                }
            },
        } as usize;
        let (Some(observation), Some(keypoint)) = (
            view.observations().get(reference),
            view.keypoint_xy(reference),
        ) else {
            return Made::Absent(Absent::NoReference);
        };
        let image = observation.image_index as usize;
        let Some((camera, pose)) = crate::bench::geometry::view_of(&recon.image_set, image) else {
            return Made::Absent(Absent::Unreadable);
        };
        let pyramid = match self.photograph(ctx, image, settled) {
            Ok(pyramid) => pyramid,
            Err(made) => return made,
        };
        let outline = crop::patch_outline(
            &patch_frame,
            &camera,
            &pose,
            [f64::from(keypoint[0]), f64::from(keypoint[1])],
            view.point().position,
            ProjectionOf::Point,
        );
        match crop::image(&outline, &pyramid) {
            Some(picture) => Made::Crop {
                picture: Box::new(picture),
                outline: Box::new(outline),
                image,
            },
            None => Made::Absent(Absent::OffImage),
        }
    }

    /// The photograph of `image` as the photograph cache holds it. While
    /// `settled`, a photograph the cache does not hold is asked for
    /// (`state::display_photograph`, which decodes it on the rayon pool and
    /// repaints when it lands); otherwise the cache is only looked in.
    fn photograph(
        &self,
        ctx: &egui::Context,
        image: usize,
        settled: bool,
    ) -> Result<Arc<ImageU8Pyramid>, Made> {
        let recon = self.node.recon();
        if settled {
            return match crate::state::display_photograph(
                &self.state.photographs,
                &self.state.photograph_requests,
                recon,
                image,
                ctx,
            ) {
                DisplayPhotograph::Decoded(pyramid) => Ok(pyramid),
                DisplayPhotograph::Decoding => {
                    self.requested.borrow_mut().push(image);
                    Err(Made::NotYet)
                }
                DisplayPhotograph::Unreadable => Err(Made::Absent(Absent::Unreadable)),
            };
        }
        let Some(path) = crate::state::photograph_path(recon, image) else {
            return Err(Made::Absent(Absent::Unreadable));
        };
        match self.state.photographs.peek_state(&path) {
            PeekedPhotograph::Decoded(pyramid) => Ok(pyramid),
            PeekedPhotograph::Unreadable => Err(Made::Absent(Absent::Unreadable)),
            PeekedPhotograph::NotDecoded => Err(Made::NotYet),
        }
    }

    /// A picture cell's hover view: the picture with what it is, or why the
    /// cell has none, and the point's portable id `id` under it.
    ///
    /// A crop shows Track View's crop hover view: the photograph's name, then
    /// the crop in three times its width and height of the photograph with the
    /// outline, the dot at the keypoint and the ring where the point projects,
    /// then the caption under it. A patch shows the picture at `size` points
    /// with the stored-patch text, or, for a point that stores no reference,
    /// a text saying that the bitmap is not the render of the photograph the
    /// Crop column shows.
    fn show_picture_hover(
        &self,
        ui: &mut egui::Ui,
        key: PictureKey,
        picture: CellPicture,
        size: f32,
        pictures: &mut PictureTextures,
        id: &str,
    ) {
        match (key.column, picture) {
            (PictureColumn::Crop, CellPicture::Texture(_)) => {
                let recon = self.node.recon();
                let pyramid = |image: usize| {
                    crate::state::peek_full_res_pyramid(&self.state.photographs, recon, image)
                };
                let image = pictures.crop(&key).map(|(_, _, image)| image);
                if let Some(name) = image
                    .and_then(|image| recon.image_set.images.get(image))
                    .map(|image| image.name.clone())
                {
                    ui.label(egui::RichText::new(name).strong());
                }
                match pictures.crop_context(ui.ctx(), key, &pyramid) {
                    Some(drawn) => drawn.show(ui),
                    None => {
                        ui.label("The photograph is being read again.");
                    }
                }
            }
            (PictureColumn::Crop, CellPicture::Absent(absent)) => {
                ui.set_max_width(320.0);
                ui.label(crop_absent_hover(absent));
            }
            (PictureColumn::Crop, CellPicture::Waiting) => {
                ui.set_max_width(320.0);
                ui.label(
                    "The photograph of the point's reference observation is being read, or its \
                     reference is being picked.",
                );
            }
            (PictureColumn::Patch, CellPicture::Texture(texture)) => {
                ui.set_max_width(size.max(240.0));
                ui.add(egui::Image::new((texture, egui::vec2(size, size))));
                // Said only where the reconstruction has a Crop column for
                // the bitmap to differ from.
                let edited = self.node.edited();
                let unreferenced = PictureColumn::Crop.is_available(edited)
                    && !edited
                        .point(key.point)
                        .and_then(|view| view.reference_observation())
                        .is_some_and(|r| r >= 0);
                ui.label(format!(
                    "{}.",
                    if unreferenced {
                        crate::patch_pictures::UNREFERENCED_PATCH_HOVER
                    } else {
                        crate::patch_pictures::STORED_PATCH_HOVER
                    }
                ));
            }
            (PictureColumn::Patch, _) => {
                ui.set_max_width(size.max(240.0));
                ui.label(format!("{}.", crate::patch_pictures::NO_STORED_PATCH_HOVER));
            }
        }
        ui.label(format!("{id}. Click to select it."));
    }

    /// Paint one row: its fill, then each cell. The pinned cells, the Point
    /// cell and the pictures, are at `pinned_left`, the viewport's left edge,
    /// and the value cells are clipped to the right of them. Answers each
    /// picture cell drawn.
    #[allow(clippy::too_many_arguments)]
    fn paint_row(
        &self,
        ui: &egui::Ui,
        rect: egui::Rect,
        pinned_left: f32,
        position: usize,
        reference: PointRef,
        pictures: &mut PictureTextures,
        settled: bool,
    ) -> Vec<DrawnPicture> {
        let point = reference.point;
        let selected = self.selected == Some(reference);
        let hovered = self.hovered == Some(reference);
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
        } else if self.stale {
            // Dimmed: the rows are of the version before an edit.
            visuals.weak_text_color()
        } else {
            visuals.text_color()
        };
        let font = egui::TextStyle::Body.resolve(ui.style());
        let row = self
            .table
            .point_indexes()
            .binary_search(&point)
            .expect("a matched point is a row of the table it was matched in");
        let clip = ui.clip_rect();
        let mut scrolled_clip = clip;
        scrolled_clip.min.x = scrolled_clip
            .min
            .x
            .max(pinned_left + self.layout.pinned_width);
        let scrolled = ui.painter().with_clip_rect(scrolled_clip);
        let mut cells = Vec::new();
        for &(slot, x, width) in &self.layout.slots {
            match slot {
                Slot::Point => {
                    ui.painter().text(
                        egui::pos2(pinned_left + CELL_PAD, rect.center().y),
                        egui::Align2::LEFT_CENTER,
                        format!("#{point}"),
                        font.clone(),
                        color,
                    );
                }
                Slot::Picture(column) => {
                    let cell = egui::Rect::from_min_size(
                        egui::pos2(pinned_left + x, rect.min.y),
                        egui::vec2(width, rect.height()),
                    );
                    let key = self.key(column, point);
                    // Made only for a row of the version on show: a row of the
                    // version before an edit may name a point the edit deleted
                    // or changed, so it draws only what was drawn for it.
                    let edited = self.node.edited();
                    let make = || match column {
                        PictureColumn::Crop => self.crop_picture(ui.ctx(), point, settled),
                        PictureColumn::Patch => patch_image(edited, point),
                    };
                    let make: Option<&dyn Fn() -> Made> = (!self.stale).then_some(&make);
                    let picture = pictures.picture(ui.ctx(), key, make);
                    let crop = pictures.crop(&key).map(|(drawn, _, _)| drawn);
                    paint_picture(
                        ui,
                        cell,
                        self.layout.picture_size,
                        picture,
                        crop,
                        self.stale,
                    );
                    cells.push((column, cell, picture));
                }
                Slot::Value(column) => {
                    scrolled.text(
                        egui::pos2(rect.min.x + x + width - CELL_PAD, rect.center().y),
                        egui::Align2::RIGHT_CENTER,
                        cell_text(self.table, column, row),
                        font.clone(),
                        color,
                    );
                }
            }
        }
        cells
    }
}

/// Whether a bitmap writer has already run the reference-view rule for the
/// point of `view` and found no eligible view, so a pick would find nothing.
///
/// Read off the point's own row, so it travels with the point through every
/// edit as its bitmap and reference do: every writer that runs the rule (the
/// open's display render, Convert to Embedded Patches, `--add-patch-bitmaps`)
/// writes a zero bitmap row and `-1` where the rule picks none. One other
/// point reads the same and is not picked: one whose reference photograph was
/// not to hand when its bitmap was rendered, so it kept its reference beside a
/// zero row, and which then lost that reference through an edit. Any other
/// point that lost its reference through an edit,
/// such as deleting the image it was in, keeps the bitmap rendered from it, so
/// it is not read as rejected and is picked again; so is a point of an older
/// file whose bitmap is a mean of its views.
fn rule_found_no_reference(view: &sfmtool_core::PointView<'_>) -> bool {
    view.reference_observation().is_none_or(|r| r < 0)
        && view
            .patch_bitmap()
            .is_some_and(|bitmap| bitmap.iter().all(|&b| b == 0))
}

/// What a *Crop* cell's hover says when the cell has no picture.
fn crop_absent_hover(absent: Absent) -> &'static str {
    match absent {
        Absent::NoPatch => {
            "The point has no patch frame, so there is no outline to cut a crop around."
        }
        Absent::Unreadable => "The photograph of the point's reference observation cannot be read.",
        Absent::NoReference => {
            "The point stores no reference observation and the reference-view rule picks none: \
             fewer than two photographs that see it can be read, or none of them sees its \
             patch well enough."
        }
        Absent::OffImage => {
            "No part of the patch's outline projects into the photograph of the point's \
             reference observation."
        }
    }
}

/// One cell's text: `…` for a column the cache does not hold yet, `-` for a
/// point with no value (the condition number and the inverse-depth z of a
/// point at infinity), a whole number for a count, `∞` for an infinite value
/// (the distance of a point at infinity), and three significant digits
/// otherwise, in scientific notation for a large condition number or
/// inverse-depth z. Positions keep up to three decimals however large they
/// are.
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
        | TrackTableColumn::RepeatObservations
        | TrackTableColumn::NormalConfidence => format!("{}", v as i64),
        TrackTableColumn::LowestZncc => format!("{v:.2}"),
        TrackTableColumn::ConditionNumber | TrackTableColumn::InverseDepthZ => {
            compact(f64::from(v))
        }
        _ => significant(f64::from(v), 3, 3),
    }
}

/// From this magnitude on, [`compact`] writes a value in scientific notation.
const SCIENTIFIC_FROM: f64 = 1e5;

/// A value that can run to many orders of magnitude, as a condition number
/// and an inverse-depth z can: three significant digits, written as `1.23e7`
/// from [`SCIENTIFIC_FROM`] on so the cell stays narrow; `∞` for an infinite
/// value.
fn compact(v: f64) -> String {
    if v.is_finite() && v.abs() >= SCIENTIFIC_FROM {
        format!("{v:.2e}")
    } else {
        significant(v, 3, 3)
    }
}

/// A position: four significant digits, with at most three
/// decimals and every whole digit kept.
fn position_text(v: f64) -> String {
    if v.is_nan() {
        return "-".to_string();
    }
    significant(v, 4, 3)
}

/// `v` to `digits` significant digits, with at most `max_decimals` decimals,
/// keeping every whole digit; `∞` or `-∞` for an infinite value.
fn significant(v: f64, digits: i32, max_decimals: i32) -> String {
    if v.is_infinite() {
        return if v > 0.0 { "\u{221e}" } else { "-\u{221e}" }.to_string();
    }
    if v.is_nan() {
        return "-".to_string();
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
