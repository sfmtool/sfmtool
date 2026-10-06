// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The observation table: the columns, one row per observation, and the
//! verdict cell each row carries.
//!
//! The image's index comes first, at the table's left edge, then the crop of
//! the photograph around the patch's outline, the rendered patch tile and the
//! verdict cell. After them come the stage's own photometric numbers, the ZNCC
//! and then the self-similarity, whose column opens with its surface plot;
//! then the reprojection error, the shift, the zoom the tile gives the
//! photograph at its centre, the kernel's status, in Edited mode where the
//! observation came from, and last the image's name.
//!
//! The verdict cell is the one column the two modes draw differently. In
//! Edited mode it is *Keep*: a switch and a pin, the row's own verdict. In
//! Viewed mode it is *Verdict*: the word `in` or `out`, the verdict the
//! read-only bars give the row, since every observation of a committed point
//! is in its track and nothing here could change that.
//!
//! Rows are painted at fixed x-offsets rather than laid out by egui, so the
//! header and every row stay aligned whatever a cell prints, and the header
//! can be drawn above the scroll area rather than as its first row: the
//! offsets are the same either side of that boundary. The one real widget in
//! a row is the *Keep* switch: the row rect is registered first and the switch
//! after it, so the switch wins the clicks that land on it and the row takes
//! the rest.
//!
//! The rows are in increasing order of *Img* until a heading is clicked. A
//! click on a heading orders them by that column, worst first where a bar
//! judges the column and increasing where none does, and a second click on
//! the same heading reverses the order; the heading the rows
//! are ordered by carries a triangle pointing up for increasing and down for
//! decreasing. A row with no reading in the column sorts after every row with
//! one, either way, and rows that tie stay in increasing order of image.
//! *Crop*, *Patch* and *From* order nothing.
//!
//! A cell with two readings, the whole patch's and its middle's, prints them
//! on two lines, each with its unit and its name (`93% whole` over `89% mid`),
//! so the headings carry names alone.
//!
//! Each reading a bar judges is drawn green when it clears the bar and red when
//! it does not, and the verdict cell is tinted by what the bars propose for the
//! row. Both come from the core's own judgement (`bar_checks` and
//! `verdicts_if_unpinned`), so the colours and the verdicts a release applies
//! cannot disagree.

use sfmtool_core::bench::{BarCheck, EditableTrack, StageKind, Thresholds, Verdict};
use sfmtool_core::patch::self_similarity::{SelfSimilarityEllipse, SelfSimilarityEllipseUnits};
use sfmtool_core::SfmrReconstruction;

use super::patch::PatchJacobian;
use super::{
    bar_box, max_self_similarity_radius, measurements, percent, provenance_text, radius_number,
    row_ellipse, row_grids, row_radius, row_surface, self_similarity_cell_color,
    self_similarity_ellipse_text, significant, zncc_cell_color, BodyMode, BoxHover, BoxesMoved,
    Judgement, RowGrids, TrackBody, TrackBodyResponse, MAX_PROJECTION_ERROR_LABEL,
    MAX_PROJECTION_ERROR_TIP, MAX_SELF_SIMILARITY_LABEL, MAX_SELF_SIMILARITY_TIP, MAX_SHIFT_LABEL,
    MAX_SHIFT_TIP, MIN_ZNCC_LABEL, MIN_ZNCC_MIDDLE_LABEL, MIN_ZNCC_MIDDLE_TIP, MIN_ZNCC_TIP,
};
use crate::scene::{ImageRef, ReconId};
use crate::state::AppState;

/// Side of one rendered tile, and so the tallest thing in a row. The size the
/// Track View draws its own tiles at, because they are the same
/// tile.
pub(crate) const TILE_SIZE: f32 = 48.0;
/// Height of one observation row.
pub(crate) const ROW_HEIGHT: f32 = TILE_SIZE + 6.0;
/// Width of the *Keep* column: the switch, then the pin that marks a verdict
/// set by hand.
const KEEP_WIDTH: f32 = 64.0;
/// Width of the *Zoom* column: room for `0.21/0.45×` with its sort triangle.
const ZOOM_WIDTH: f32 = 76.0;
/// Width of the *From* column: room for its longest cell, `feature 123456`.
const FROM_WIDTH: f32 = 110.0;
/// Width of the *Name* column, the last one. A name longer than this is
/// elided in its middle, and hovering it or the *Img* cell shows it whole.
const NAME_WIDTH: f32 = 220.0;
/// Width of the part of the *Keep* cell the switch takes; the pin takes the
/// rest.
const SWITCH_CELL_WIDTH: f32 = 40.0;
/// Size of the *Keep* switch.
const SWITCH_SIZE: egui::Vec2 = egui::vec2(34.0, 18.0);
/// The fill of an enabled *Keep* switch that is on. A greyed switch that is
/// on is filled grey instead.
pub(super) const KEEP_ON_FILL: egui::Color32 = egui::Color32::from_rgb(56, 150, 76);
/// Side of one cell of a three-by-three grid a row draws: room for the line
/// along its ellipse's major axis the self-similarity grid draws in a cell.
const GRID_CELL: f32 = 10.0;
/// Side of a whole grid: three cells and the four lines of its border.
const GRID_SIDE: f32 = 3.0 * GRID_CELL + 4.0;
/// Side of the self-similarity surface plot a row draws.
const PLOT_SIDE: f32 = 44.0;
/// Side of the same plot in its hover view.
const PLOT_HOVER_SIDE: f32 = 220.0;
/// Half the width of the triangle after the heading the rows are ordered by.
/// Small, since *Img* has little room before *Crop*.
const SORT_TRIANGLE_HALF_WIDTH: f32 = 4.0;

/// What one row drew, kept so that a test reads the table the app draws rather
/// than a second computation of it.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct RowSummary {
    /// The observation's index in the track, which is stable for its life.
    pub observation: usize,
    /// The image it names.
    pub image: u32,
    /// The verdict it carries.
    pub verdict: Verdict,
    /// Whether that verdict was set by hand.
    pub pinned: bool,
    /// What the thresholds propose for it were its verdict unpinned, which is
    /// what its verdict cell is tinted by, or `None` for a row nothing at this
    /// stage has measured, whose cell is not tinted. In Viewed mode this is
    /// the row's *Verdict*.
    pub proposal: Option<Verdict>,
    /// The verdict cell's hover text: in Edited mode what the switch and the
    /// pin say and why the bars propose what they do, in Viewed mode why the
    /// bars give the row the verdict they do.
    pub keep_hover: String,
    /// The *Verdict* cell as printed, `in`, `out`, `out (2)` or `-`, in Viewed mode;
    /// `None` in Edited mode, which draws a switch there.
    pub verdict_text: Option<String>,
    /// The colour the verdict cell was tinted, or `None` for an untinted one.
    pub tint: Option<egui::Color32>,
    /// The line the crop's hover view adds under the picture: the pixel and
    /// the feature index. `None` where no crop was drawn.
    pub crop_caption: Option<String>,
    /// The five measurement cells, as printed, a cell with two readings
    /// holding them on two lines.
    pub cells: [String; 5],
    /// The whole and middle readings' ellipses, which the *Self-similarity*
    /// cell's hover lays out under [`SELF_SIMILARITY_ELLIPSE_CAPTION`] in grid
    /// px, image px and along the patch's axes
    /// ([`super::self_similarity_ellipse_text`], built only while the cell is
    /// hovered). Both `None` where the cell has no hover.
    pub self_similarity_ellipse: [Option<SelfSimilarityEllipseUnits>; 2],
    /// The Jacobian at the centre of the row's tile, in photograph pixels per
    /// patch-grid px at the reconstruction's patch resolution, computed
    /// without the photograph ([`super::tile::patch_jacobian`], which lists
    /// where there is none).
    pub jacobian: Option<PatchJacobian>,
    /// The *Zoom* cell as printed, `3.1/4.8×`, or `-`.
    pub zoom_text: String,
    /// What each line of each cell was coloured by, indexed as
    /// [`RowSummary::cells`] and then by line: a pass is drawn green, a fail
    /// red, and a reading no bar judged in the plain text colour. A cell's
    /// second entry is [`BarCheck::NotJudged`] where it has one line.
    pub checks: [[BarCheck; 2]; 5],
    /// The two grids drawn beside the ZNCC and the self-similarity cells.
    pub grids: RowGrids,
    /// Whether the row drew a rendered tile, rather than the empty frame that
    /// stands in when there is nothing to render.
    pub tile: bool,
    /// Whether the row drew the crop of its photograph around the patch's
    /// outline, rather than the empty frame that stands in for it.
    pub crop: bool,
    /// Whether the row drew the self-similarity surface plot.
    pub self_similarity_plot: bool,
}

/// Fixed column x-offsets, relative to the left edge of the table.
pub(super) struct ColumnLayout {
    image: f32,
    crop: f32,
    tile: f32,
    keep: f32,
    zncc: f32,
    zncc_grid: f32,
    self_similarity_plot: f32,
    self_similarity: f32,
    self_similarity_grid: f32,
    offset: f32,
    shift: f32,
    zoom: f32,
    status: f32,
    from: f32,
}

impl ColumnLayout {
    pub(super) fn new() -> Self {
        let image = 0.0;
        // Room for the *Img* heading and the sort triangle after it.
        let crop = image + 44.0;
        let tile = crop + TILE_SIZE + 4.0;
        let keep = tile + TILE_SIZE + 8.0;
        let zncc = keep + KEEP_WIDTH + 6.0;
        // Room for `100% whole`, then the ZNCC grid.
        let zncc_grid = zncc + 80.0;
        // The self-similarity column opens with the whole tile's surface plot.
        let self_similarity_plot = zncc_grid + GRID_SIDE + 10.0;
        // Room for `2.3 px whole`, then the self-similarity grid.
        let self_similarity = self_similarity_plot + PLOT_SIDE + 8.0;
        let self_similarity_grid = self_similarity + 88.0;
        let offset = self_similarity_grid + GRID_SIDE + 10.0;
        // Room for the error in px over the same residual in degrees,
        // `12.65 px` over `0.08°`, and for the bar's box and its `px` in the
        // threshold row.
        let shift = offset + 76.0;
        // Room for `12.25 px`. The tile's zoom follows it.
        let zoom = shift + 62.0;
        let status = zoom + ZOOM_WIDTH;
        // The status cell holds a sentence at the track stage -- the reason a
        // row was not read, or the walk a fit refused and what it scored -- so
        // it is given room for one and elided to it.
        let from = status + 270.0;
        Self {
            image,
            crop,
            tile,
            keep,
            zncc,
            zncc_grid,
            self_similarity_plot,
            self_similarity,
            self_similarity_grid,
            offset,
            shift,
            zoom,
            status,
            from,
        }
    }

    /// The table's width in `mode`, to the end of *Name*. What the table
    /// scrolls sideways over when the panel is narrower.
    pub(super) fn width(&self, mode: BodyMode) -> f32 {
        self.name_x(mode) + NAME_WIDTH
    }

    /// The *Name* column's offset from the table's left edge in `mode`. It is
    /// the last column, after *From* in Edited mode and after *Status* in
    /// Viewed mode, which draws no *From*.
    pub(super) fn name_x(&self, mode: BodyMode) -> f32 {
        match mode {
            BodyMode::Edited => self.from + FROM_WIDTH,
            BodyMode::Viewed => self.from,
        }
    }

    /// The *Img* column's offset from the table's left edge.
    #[cfg(test)]
    pub(super) fn image_x(&self) -> f32 {
        self.image
    }

    /// The tile column's offset from the table's left edge.
    #[cfg(test)]
    pub(super) fn tile_x(&self) -> f32 {
        self.tile
    }

    /// The crop column's offset from the table's left edge.
    #[cfg(test)]
    pub(super) fn crop_x(&self) -> f32 {
        self.crop
    }

    /// The *Keep* column's offset from the table's left edge.
    #[cfg(test)]
    pub(super) fn keep_x(&self) -> f32 {
        self.keep
    }

    /// The header's cells for `mode`, each at the offset its column is drawn
    /// at, with the hover text that says what the column holds: *Keep* and
    /// *From* in Edited mode, *Verdict* in *Keep*'s place and no *From* in
    /// Viewed mode.
    pub(super) fn headers(&self, mode: BodyMode) -> Vec<(f32, &'static str, &'static str)> {
        let verdict = match mode {
            BodyMode::Edited => (self.keep, "Keep", KEEP_TIP),
            BodyMode::Viewed => (self.keep, "Verdict", VERDICT_TIP),
        };
        let mut headers = vec![
            (
                self.image,
                "Img",
                "The image's index in the reconstruction.",
            ),
            (self.crop, "Crop", CROP_TIP),
            (self.tile, "Patch", PATCH_TIP),
            verdict,
            (self.zncc, "ZNCC", ZNCC_TIP),
            (
                self.self_similarity_plot,
                "Self-similarity",
                SELF_SIMILARITY_TIP,
            ),
            (self.offset, "Proj. err", PROJECTION_ERROR_TIP),
            (self.shift, "Shift", SHIFT_TIP),
            (self.zoom, "Zoom", ZOOM_TIP),
            (self.status, "Status", STATUS_TIP),
        ];
        if mode == BodyMode::Edited {
            headers.push((self.from, "From", FROM_TIP));
        }
        headers.push((self.name_x(mode), "Name", "The image's file name."));
        headers
    }
}

/// A column the rows can be put in order by.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SortColumn {
    /// The image's index.
    Image,
    /// How many bars the row fails: *Keep* in Edited mode and *Verdict* in
    /// Viewed mode. Among rows that fail the same number, the ones the bars
    /// propose `in` come before the ones they propose `out`, which for a row
    /// that fails none is one whose image another sighting holds.
    Verdict,
    /// The whole patch's ZNCC.
    Zncc,
    /// The whole tile's self-similarity radius.
    SelfSimilarity,
    /// The reprojection error in px.
    ProjectionError,
    /// The shift in px.
    Shift,
    /// The zoom the tile applies to the photograph, as the geometric mean
    /// over its two singular directions.
    Zoom,
    /// The status cell's text.
    Status,
    /// The image's name.
    Name,
}

impl SortColumn {
    /// The column the heading `heading` orders the rows by, or `None` for
    /// one that orders nothing: *Crop* and *Patch*, which are pictures, and
    /// *From*.
    pub(super) fn of_heading(heading: &str) -> Option<Self> {
        Some(match heading {
            "Img" => SortColumn::Image,
            "Keep" | "Verdict" => SortColumn::Verdict,
            "ZNCC" => SortColumn::Zncc,
            "Self-similarity" => SortColumn::SelfSimilarity,
            "Proj. err" => SortColumn::ProjectionError,
            "Shift" => SortColumn::Shift,
            "Zoom" => SortColumn::Zoom,
            "Status" => SortColumn::Status,
            "Name" => SortColumn::Name,
            _ => return None,
        })
    }

    /// Whether a click that makes this the column the rows are ordered by
    /// orders them decreasing. A column a bar judges starts at its worst
    /// rows: the most bars failed, the lowest ZNCC, the largest
    /// self-similarity radius, projection error and shift. A column no bar
    /// judges starts increasing. *Zoom* is one of those: no bar says one zoom
    /// is worse than another.
    pub(super) fn worst_first_descending(self) -> bool {
        match self {
            SortColumn::Verdict
            | SortColumn::SelfSimilarity
            | SortColumn::ProjectionError
            | SortColumn::Shift => true,
            SortColumn::Image
            | SortColumn::Zncc
            | SortColumn::Zoom
            | SortColumn::Status
            | SortColumn::Name => false,
        }
    }

    /// What the heading's accessible name calls the column.
    fn name(self) -> &'static str {
        match self {
            SortColumn::Image => "image",
            SortColumn::Verdict => "missed thresholds",
            SortColumn::Zncc => "ZNCC",
            SortColumn::SelfSimilarity => "self-similarity",
            SortColumn::ProjectionError => "projection error",
            SortColumn::Shift => "shift",
            SortColumn::Zoom => "zoom",
            SortColumn::Status => "status",
            SortColumn::Name => "name",
        }
    }
}

/// The order the rows are drawn in: by which column, and which way. A tool
/// setting kept for the session, as *Lock* is: it moves nothing on the track.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct TableSort {
    /// The column the rows are ordered by.
    pub column: SortColumn,
    /// Whether the order is decreasing rather than increasing.
    pub descending: bool,
}

impl Default for TableSort {
    /// Increasing by image.
    fn default() -> Self {
        Self {
            column: SortColumn::Image,
            descending: false,
        }
    }
}

impl TableSort {
    /// The order a click on `column`'s heading leaves: the same column the
    /// other way when the rows are already ordered by it, and otherwise that
    /// column in the direction [`SortColumn::worst_first_descending`] gives.
    pub(super) fn clicked(self, column: SortColumn) -> Self {
        let descending = if self.column == column {
            !self.descending
        } else {
            column.worst_first_descending()
        };
        Self { column, descending }
    }
}

/// What a row is ordered by in one column: two numbers compared in turn, or
/// a text. Only a verdict uses the second number, `1` for a row proposed
/// `out` and `0` for one proposed `in`.
#[derive(Debug, Clone, PartialEq)]
pub(super) enum SortKey {
    Number(f64, f64),
    Text(String),
}

impl SortKey {
    fn compare(&self, other: &Self) -> std::cmp::Ordering {
        match (self, other) {
            (SortKey::Number(a0, a1), SortKey::Number(b0, b1)) => {
                a0.total_cmp(b0).then(a1.total_cmp(b1))
            }
            (SortKey::Text(a), SortKey::Text(b)) => a.cmp(b),
            // One column gives every row the same kind of key.
            _ => std::cmp::Ordering::Equal,
        }
    }
}

/// One reading of a cluster-stage measurement, which a column orders by.
type ClusterReading = fn(&sfmtool_core::bench::ClusterMeasurement) -> Option<f64>;

/// One reading of a track-stage measurement, which a column orders by.
type TrackReading = fn(&sfmtool_core::bench::TrackMeasurement) -> Option<f64>;

/// A reading as a key: none for a missing or `NaN` reading.
fn number_key(value: Option<f64>) -> Option<SortKey> {
    value
        .filter(|v| !v.is_nan())
        .map(|v| SortKey::Number(v, 0.0))
}

/// The order to draw rows in, as indexes into `keys`: by key, increasing or
/// decreasing as `sort` says, with every row that has no key after every row
/// that has one. Ties keep increasing order of `images`, then of index.
pub(super) fn sorted_rows(keys: &[Option<SortKey>], images: &[u32], sort: TableSort) -> Vec<usize> {
    let mut order: Vec<usize> = (0..keys.len()).collect();
    order.sort_by(|&a, &b| {
        let by_key = match (&keys[a], &keys[b]) {
            (Some(ka), Some(kb)) => {
                let ord = ka.compare(kb);
                if sort.descending {
                    ord.reverse()
                } else {
                    ord
                }
            }
            (Some(_), None) => std::cmp::Ordering::Less,
            (None, Some(_)) => std::cmp::Ordering::Greater,
            (None, None) => std::cmp::Ordering::Equal,
        };
        by_key.then(images[a].cmp(&images[b])).then(a.cmp(&b))
    });
    order
}

/// The *Verdict* cell's text for a proposal and its checks: `in`, `out`, or
/// `-` where nothing has measured the row. An `out` that fails bars says how
/// many, `out (2)`; one that fails none lost its image to another sighting.
pub(super) fn verdict_text(judged: Option<&Judgement>) -> String {
    match judged {
        None => "-".to_string(),
        Some(j) if j.proposal == Verdict::In => "in".to_string(),
        Some(j) => match j.checks.failed() {
            0 => "out".to_string(),
            n => format!("out ({n})"),
        },
    }
}

/// The *Zoom* cell's text: the range of patch-grid px, at the
/// reconstruction's patch resolution, per photograph pixel over the two
/// singular directions, least over most, `3.1/4.8×`, both numbers even
/// where the two print the same, `0.51/0.51×`, or `-` where there is no
/// reading. Each number carries two significant digits, judged after rounding
/// ([`super::significant`]), so `0.031` prints `0.031`, `9.96` prints `10` and
/// `0.996` prints `1.0`; a zoom of 100 or more prints whole.
pub(super) fn zoom_text(jacobian: Option<PatchJacobian>) -> String {
    let Some([low, high]) = jacobian.and_then(|j| j.zoom_range()) else {
        return "-".to_string();
    };
    format!("{}/{}\u{d7}", significant(low), significant(high))
}

/// The *Zoom* cell's hover text: the sampler the row's tile is rendered with,
/// and the loss along the less compressed axis the sampler rule chose it by.
pub(super) fn zoom_sampler_text(jacobian: &PatchJacobian) -> String {
    let loss = jacobian.minor_axis_loss();
    let loss = if loss.is_finite() {
        format!("{loss:.2}")
    } else {
        "-".to_string()
    };
    match jacobian.sampler() {
        sfmtool_core::camera::sampler::Sampler::Anisotropic => format!(
            "Rendered with the anisotropic sampler: one mip level for both axes would read \
             the less compressed axis {loss}\u{d7} too coarsely."
        ),
        _ => format!(
            "Rendered with the bilinear_mip sampler: one mip level for both axes reads the \
             less compressed axis {loss}\u{d7} too coarsely, under the {}\u{d7} that moves \
             a view to the anisotropic sampler, or the view is not compressed enough for \
             the level to matter.",
            sfmtool_core::camera::sampler::DEFAULT_ANISOTROPIC_THRESHOLD
        ),
    }
}

/// The *Verdict* heading's hover text, in Viewed mode.
pub(super) const VERDICT_TIP: &str = "What the thresholds say about the observation: in where \
    it clears every bar and holds its image, out where it does not, and - where nothing has \
    measured it yet. It is the verdict the bench's own evaluation would give the row once the \
    point is on the bench and the row unpinned.\n\n\
    Every observation of a committed point is in its track, and nothing here changes that: \
    the threshold boxes recolour this column and the readings. Tick Edit to work on the \
    track. Hover a cell for the reason.\n\n\
    An out row that fails bars says how many in brackets, out (2); one that fails none lost \
    its image to another sighting. Click the heading to order the rows by how many bars they \
    fail.";

/// The *Keep* heading's hover text.
pub(super) const KEEP_TIP: &str = "Whether the track keeps the observation. A kept observation \
    is one the evaluation and a fit read the track by, and one a commit writes.\n\n\
    The thresholds set the switch of every unpinned row each time the track is evaluated and \
    when a threshold box is let go. Click a switch to set it by hand, which pins it. The pin \
    beside the switch is solid on a pinned verdict, which the thresholds leave alone, and a \
    faint outline on one they set. Click the pin to unpin a verdict and let the thresholds \
    decide again, or to pin one as it stands. The pin in this heading unpins every pinned \
    verdict of the track at once, and when none is pinned it pins every verdict as it \
    stands.\n\n\
    The cell is green when the bars propose keeping the observation and red when they \
    propose turning it out, as they would were its verdict unpinned, so a switch that is on \
    in a red cell is a hand ruling against the bars. Hover a switch for the reason.";

/// The *Crop* heading's hover text.
pub(super) const CROP_TIP: &str = "The photograph itself around the patch's outline as it \
    lands there, the outline Image Detail draws, in a square with the outline in the middle. \
    It shows how the lens and the view bend the patch, which the square patch beside it \
    hides.\n\n\
    Hover a crop to see it in three times its width of the photograph, with a dot where the \
    observation sits, a ring where the track's point projects, the patch's two axes in the \
    photograph's pixels, and the pixel the observation sits at with its feature index.";

/// The *Patch* heading's hover text.
pub(super) const PATCH_TIP: &str = "The patch as this photograph sees it, warped square: at \
    the track stage the patch re-rendered from this photograph where the observation sits, \
    at the cluster stage the grid the refinement samples there. It is the picture the ZNCC \
    and self-similarity numbers are read from.\n\n\
    Hover a patch to see it in three times its width of the photograph, with a dot where the \
    observation sits and a ring where the track's point projects.";

/// The ZNCC heading's hover text, in one constant so the tests aim at the text
/// shown.
pub(super) const ZNCC_TIP: &str = "Zero-mean normalized cross-correlation, in percent: how \
    closely this photograph's view of the patch matches, once differences in brightness and \
    contrast are removed. 100% is an exact match and 0% is no relation.\n\n\
    whole is over the whole patch. mid is over its middle only, the centred square half the \
    patch's width, read from the same samples. A high whole with a low mid means the match \
    comes from the patch's surroundings rather than from the pixel's own neighbourhood.\n\n\
    The grid beside them is the same samples read over each ninth of the patch, laid out as the \
    tile is, with every pixel weighted equally: green at 100, yellow at 75, red at 50 and \
    below, grey where the patch is flat. Hover it for the numbers.\n\n\
    At the cluster stage the match is against the reference's template. At the track stage it \
    is against the consensus of the other observations, with this one left out.";

/// The shift heading's hover text.
pub(super) const SHIFT_TIP: &str = "How far the correlation peak sits from where the \
    observation sits, in patch-grid pixels, the unit of the self-similarity radius: the \
    observation's own evidence of where it belongs.\n\n\
    At the track stage the evaluation looks for the peak within the shift bar of the \
    sighting, against the consensus of the others, and moves nothing. A shift inside the \
    self-similarity radius is within what the patch cannot tell apart; one beyond it says the \
    other photographs want the sighting moved. A fit moves it, up to the bar.\n\n\
    At the cluster stage it is how far the refinement moved the member off its seed.\n\n\
    The box under this heading is the shift bar, which judges it.";

/// The *Zoom* heading's hover text.
pub(super) const ZOOM_TIP: &str = "How much the patch magnifies the photograph at its \
    centre: patch-grid px per pixel of the photograph, read from the Jacobian of the warp \
    from the patch to the photograph. The grid is the reconstruction's own patch resolution, \
    the one its patch bitmaps are stored at (24 px a side unless the file says otherwise), \
    and the unit of the shift and the self-similarity radius; it is not the resolution the \
    tile here is drawn at. Over 1\u{d7} the patch samples the photograph more finely than \
    its pixels, and under 1\u{d7} more coarsely. It is geometry alone, read from the patch, \
    the camera, the image's pose and where the observation sits, so it does not wait for the \
    photograph, and a tile whose middle is off the photograph still has one.\n\n\
    The warp can stretch one direction more than another, so the cell gives the least zoom \
    over the most, 0.71/1.3\u{d7}, and gives both even where the two agree, \
    0.51/0.51\u{d7}. Each carries two significant digits.\n\n\
    The zoom also decides how the tile is drawn, as the bench's own renders decide it: \
    where one mip level for both axes would read the less compressed axis at least 1.5\u{d7} \
    too coarsely, the tile is rendered with the anisotropic sampler. Hover a cell to see \
    which.\n\n\
    A - is a row at the cluster stage, whose tile is not drawn through a warp; a track with \
    no patch yet; an observation with nothing saying where it sits; a patch whose centre is \
    behind the camera or outside the camera model's domain; or a patch seen edge on.\n\n\
    Ordering by this column orders by the geometric mean of the two.";

/// The projection error heading's hover text.
pub(super) const PROJECTION_ERROR_TIP: &str = "The reprojection error, in pixels over the \
    same residual in degrees.\n\n\
    The pixels are how far the observation sits from where the track's point projects into \
    this image. Before the track is triangulated it is measured to where its patch's centre \
    projects, which is the same place once it is.\n\n\
    The degrees are the angle between the observation's ray and the direction from its camera \
    to the point. They are comparable across lenses and depths, where a pixel is not.\n\n\
    Large errors on every row beside small seed shifts say the point is off, not the \
    sightings. Track stage only.\n\n\
    The box under this heading is the bar that judges the px.";

/// The self-similarity heading's hover text.
pub(super) const SELF_SIMILARITY_TIP: &str = "The ZNCC self-similarity radius: how far, in \
    patch-grid pixels, this observation's own tile can slide over itself and still match \
    itself as well as a true match between two photographs would: the semi-major axis of the \
    ellipse fitted to the shifts where its ZNCC against itself, interpolated between \
    whole-pixel shifts, stays at or above that level. \
    Under 1 means a match locks onto this position within a pixel, as on a corner or a busy \
    texture. 3+ means it still matched itself 3 pixels away and may slide further, as along a \
    straight edge or over a flat patch.\n\n\
    whole is the whole tile, mid its middle alone, the centred square half its width. The \
    grid beside them is each ninth of the tile alone, laid out as the tile is: \
    green under 1, yellow from 1 to 2, orange from 2 to 3, red at 3 or more. A line in a box is \
    the direction that ninth can slide in, along its ellipse's major axis, where the ellipse \
    is long and thin. Hover the grid for the numbers, and the cell for the ellipses.\n\n\
    The box under this heading is the bar that judges the whole tile's radius.";

const STATUS_TIP: &str = "What the last evaluation or fit said about the row. At the \
    cluster stage, the refinement's verdict on the member. At the track stage, localized, a \
    walk the fit refused, or why the row could not be read.";

const FROM_TIP: &str = "Where the observation came from: the point the track was read \
    from, a detected feature, a descriptor search, a sweep, a pixel placed by hand, or another \
    point.";

/// The label a pinned verdict's menu entry carries.
pub(super) const UNPIN_LABEL: &str = "Unpin, let the thresholds decide";

/// The row menu's unpin entry when the row is one of several selected, which
/// unpins every pinned verdict among them in one step.
pub(super) fn unpin_selection_label(count: usize) -> String {
    let noun = if count == 1 { "verdict" } else { "verdicts" };
    format!("Unpin {count} {noun}, let the thresholds decide")
}

/// The accessible name of the *Keep* heading's pin, which says what a click
/// does: unpin every pinned row when any is pinned, pin every row when none is.
pub(super) fn heading_pin_name(pinned: usize) -> &'static str {
    if pinned > 0 {
        "Unpin all"
    } else {
        "Pin all"
    }
}

/// The *Keep* heading's pin's hover text: what a click does, with the count,
/// or why there is nothing to do. `pinned` is how many rows are pinned and
/// `rows` how many the track has.
pub(super) fn heading_pin_hover(pinned: usize, rows: usize, busy: Option<&str>) -> String {
    match (pinned, rows, busy) {
        (_, 0, _) => "The track has no observations to pin.".to_string(),
        (_, _, Some(why)) => why.to_string(),
        (1, _, None) => "Unpin the 1 pinned verdict and let the bars decide".to_string(),
        (0, 1, None) => "Pin the 1 verdict as it stands".to_string(),
        (0, n, None) => format!("Pin all {n} verdicts as they stand"),
        (n, _, None) => format!("Unpin all {n} pinned verdicts and let the bars decide"),
    }
}

/// The *Keep* switch of one row, filling `rect`. The whole of `rect` takes
/// the click, so the target is the cell and not the switch's own few points.
/// It takes a drag too, which does nothing, so a drag begun on a switch does
/// not scroll the table. Returns the click's response.
///
/// When not `enabled` it still takes the click and the drag, and the caller
/// does nothing with them, so a click on a refused switch is not a click on
/// the row. It is drawn greyed: grey where an enabled switch is green, with
/// the outline of a widget that takes no input, and the whole at the opacity
/// egui draws a disabled widget at.
fn keep_switch(
    ui: &mut egui::Ui,
    rect: egui::Rect,
    id: egui::Id,
    kept: bool,
    enabled: bool,
) -> egui::Response {
    let response = ui.interact(rect, id, egui::Sense::click_and_drag());
    response.widget_info(|| {
        egui::WidgetInfo::selected(
            egui::WidgetType::Checkbox,
            enabled && ui.is_enabled(),
            kept,
            "Keep",
        )
    });
    if !ui.is_rect_visible(rect) {
        return response;
    }
    let switch = egui::Rect::from_min_size(
        egui::pos2(rect.min.x + 2.0, rect.center().y - SWITCH_SIZE.y / 2.0),
        SWITCH_SIZE,
    );
    let visuals = if enabled {
        ui.style().interact_selectable(&response, kept)
    } else {
        ui.visuals().widgets.noninteractive
    };
    let how_on = ui.ctx().animate_bool_responsive(id, kept);
    let radius = 0.5 * switch.height();
    let track_fill = if kept && enabled {
        KEEP_ON_FILL
    } else if kept {
        ui.visuals().weak_text_color()
    } else {
        ui.visuals().widgets.inactive.bg_fill
    };
    let painter = switch_painter(ui, enabled);
    painter.rect(
        switch,
        radius,
        track_fill,
        visuals.bg_stroke,
        egui::StrokeKind::Inside,
    );
    let knob_x = egui::lerp((switch.left() + radius)..=(switch.right() - radius), how_on);
    painter.circle(
        egui::pos2(knob_x, switch.center().y),
        0.75 * radius,
        ui.visuals().strong_text_color(),
        egui::Stroke::NONE,
    );
    response
}

/// A row's *Verdict* cell, in Viewed mode: the word `in` or `out` for the
/// verdict the read-only bars give the row, with how many bars an `out` row
/// fails, or `-` where nothing has measured it, in the cell the caller has
/// tinted. The cell takes the pointer for its
/// hover text and no click, so a click on it is still the row's. Returns the
/// hover text and the word.
fn draw_verdict(
    ui: &mut egui::Ui,
    rect: egui::Rect,
    cols: &ColumnLayout,
    observation: usize,
    judged: Option<&Judgement>,
    image: u32,
) -> (String, String) {
    let cell = egui::Rect::from_min_max(
        egui::pos2(rect.min.x + cols.keep, rect.min.y),
        egui::pos2(rect.min.x + cols.keep + KEEP_WIDTH, rect.max.y),
    );
    let text = verdict_text(judged);
    ui.painter().text(
        egui::pos2(cell.min.x + 6.0, cell.center().y),
        egui::Align2::LEFT_CENTER,
        &text,
        egui::TextStyle::Body.resolve(ui.style()),
        ui.visuals().text_color(),
    );
    let hover = proposal_reason(judged, image);
    ui.interact(
        cell,
        ui.id().with(("track_view_verdict", observation)),
        egui::Sense::hover(),
    )
    .on_hover_text(&hover);
    (hover, text)
}

/// The pin beside a row's *Keep* switch, filling `rect`: a pushpin drawn solid
/// when the verdict was set by hand and as a faint outline when the thresholds
/// set it. The whole of `rect` takes the click, and a drag, as the switch
/// does. Returns the click's response.
///
/// When not `enabled` it still takes the click and the drag, as the switch
/// does, does not brighten under the pointer, and is drawn at egui's disabled
/// opacity.
fn pin_toggle(
    ui: &mut egui::Ui,
    rect: egui::Rect,
    id: egui::Id,
    pinned: bool,
    enabled: bool,
) -> egui::Response {
    let response = ui.interact(rect, id, egui::Sense::click_and_drag());
    response.widget_info(|| {
        egui::WidgetInfo::selected(
            egui::WidgetType::Checkbox,
            enabled && ui.is_enabled(),
            pinned,
            "Pin",
        )
    });
    if !ui.is_rect_visible(rect) {
        return response;
    }
    let visuals = ui.visuals();
    let color = match (pinned, enabled && response.hovered()) {
        (true, _) => visuals.strong_text_color(),
        (false, true) => visuals.text_color(),
        (false, false) => visuals.weak_text_color().gamma_multiply(0.6),
    };
    paint_pushpin(&switch_painter(ui, enabled), rect.center(), color, pinned);
    response
}

/// The painter a row's switch and pin draw with: the `ui`'s own, faded to
/// egui's disabled opacity when not `enabled`, so they look greyed as the
/// widgets egui disables do.
fn switch_painter(ui: &egui::Ui, enabled: bool) -> egui::Painter {
    let mut painter = ui.painter().clone();
    if !enabled {
        painter.multiply_opacity(ui.visuals().disabled_alpha());
    }
    painter
}

/// A pushpin standing upright at `c`: a cap, a body narrower than the cap, a
/// collar and the needle below it. Filled when `solid`, outlined otherwise.
fn paint_pushpin(painter: &egui::Painter, c: egui::Pos2, color: egui::Color32, solid: bool) {
    let stroke = egui::Stroke::new(1.2, color);
    let cap = egui::Rect::from_center_size(c + egui::vec2(0.0, -6.0), egui::vec2(8.0, 3.0));
    let body = egui::Rect::from_min_max(
        egui::pos2(c.x - 2.5, cap.bottom()),
        egui::pos2(c.x + 2.5, c.y + 1.5),
    );
    let collar = [
        egui::pos2(c.x - 5.5, c.y + 1.5),
        egui::pos2(c.x + 5.5, c.y + 1.5),
    ];
    for part in [cap, body] {
        if solid {
            painter.rect_filled(part, 1.0, color);
        } else {
            painter.rect_stroke(part, 1.0, stroke, egui::StrokeKind::Inside);
        }
    }
    painter.line_segment(collar, egui::Stroke::new(1.6, color));
    painter.line_segment(
        [egui::pos2(c.x, c.y + 1.5), egui::pos2(c.x, c.y + 8.0)],
        stroke,
    );
}

/// The switch's hover text: what the switch says now, how to change it, and
/// why the bars propose what they do for the row.
fn keep_hover(kept: bool, pinned: bool, judged: Option<&Judgement>, image: u32) -> String {
    let state = match (kept, pinned) {
        (true, true) => "Kept, set by hand.",
        (true, false) => "Kept, as the thresholds propose.",
        (false, true) => "Not kept, set by hand.",
        (false, false) => {
            "Not kept: the thresholds did not take it, or nothing has measured it yet."
        }
    };
    format!(
        "{state} Click to switch it, which pins it.\n\n{}",
        proposal_reason(judged, image)
    )
}

/// Why the bars propose what they do for a row in `image`: which bars it
/// fails, named as the column headings name the readings, or, for a row that
/// clears every bar and is still proposed `out`, that another sighting of its
/// image keeps the image's one `in`.
fn proposal_reason(judged: Option<&Judgement>, image: u32) -> String {
    let Some(judged) = judged else {
        return "Nothing at this stage has measured it, so the bars propose nothing.".to_string();
    };
    if judged.proposal == Verdict::In {
        return "The bars propose keeping it: it clears every bar.".to_string();
    }
    let checks = &judged.checks;
    let failed: Vec<&str> = [
        (checks.min_zncc, "ZNCC whole is under the bar"),
        (checks.min_zncc_middle, "ZNCC mid is under the bar"),
        (checks.max_shift_px, "Shift is over the bar"),
        (
            checks.max_zncc_self_similarity_radius,
            "Self-similarity whole is over the bar",
        ),
        (checks.max_projection_error_px, "Proj. err is over the bar"),
    ]
    .into_iter()
    .filter(|&(check, _)| check == BarCheck::Fail)
    .map(|(_, why)| why)
    .collect();
    if failed.is_empty() {
        format!(
            "The bars propose turning it out: it clears every bar, but another sighting in \
             image {image} is kept, and a track keeps one sighting per image."
        )
    } else {
        format!("The bars propose turning it out: {}.", failed.join("; "))
    }
}

/// The text colour a reading is drawn in by what its bar says of it: a green
/// and a red chosen to read as text on the panel's background in the dark and
/// the light visuals, and `plain` for a reading no bar judged.
fn check_color(visuals: &egui::Visuals, check: BarCheck, plain: egui::Color32) -> egui::Color32 {
    match (check, visuals.dark_mode) {
        (BarCheck::NotJudged, _) => plain,
        (BarCheck::Pass, true) => egui::Color32::from_rgb(110, 205, 125),
        (BarCheck::Fail, true) => egui::Color32::from_rgb(240, 115, 105),
        (BarCheck::Pass, false) => egui::Color32::from_rgb(25, 125, 50),
        (BarCheck::Fail, false) => egui::Color32::from_rgb(195, 40, 35),
    }
}

/// The tint of a row's *Keep* cell by what the bars propose for the row, or
/// `None` for a row they propose nothing for.
pub(super) fn proposal_tint(
    visuals: &egui::Visuals,
    proposal: Option<Verdict>,
) -> Option<egui::Color32> {
    Some(match (proposal?, visuals.dark_mode) {
        (Verdict::In, true) => egui::Color32::from_rgb(34, 78, 44),
        (Verdict::Out, true) => egui::Color32::from_rgb(88, 40, 40),
        (Verdict::In, false) => egui::Color32::from_rgb(196, 232, 200),
        (Verdict::Out, false) => egui::Color32::from_rgb(244, 200, 196),
    })
}

/// What each line of a row's five cells is coloured by: the bar that judges
/// it, from `judged`, where the cell prints the reading that bar judged. No
/// line is judged on a row the bars do not judge or whose cells print no
/// numbers. The projection error's bar judges its px line, and nothing judges
/// the degrees under it.
pub(super) fn cell_checks(
    judged: Option<&Judgement>,
    evaluation: &crate::bench::live::Evaluation,
) -> [[BarCheck; 2]; 5] {
    use crate::bench::live::Evaluation;
    let none = [BarCheck::NotJudged; 2];
    let Some(judged) = judged else {
        return [none; 5];
    };
    if matches!(evaluation, Evaluation::Refused(_) | Evaluation::Failed(_)) {
        return [none; 5];
    }
    let checks = &judged.checks;
    [
        [checks.min_zncc, checks.min_zncc_middle],
        [checks.max_shift_px, BarCheck::NotJudged],
        [checks.max_projection_error_px, BarCheck::NotJudged],
        [checks.max_zncc_self_similarity_radius, BarCheck::NotJudged],
        none,
    ]
}

/// The pin's hover text: what it says now, and what a click does.
fn pin_hover(pinned: bool) -> &'static str {
    if pinned {
        "Pinned: this verdict was set by hand, and the thresholds leave it alone. Click to \
         unpin it and let the thresholds decide."
    } else {
        "Not pinned: the thresholds set this verdict, and move it when a threshold moves. \
         Click to pin it as it stands."
    }
}

/// The menu entry that hands pinned verdicts back to the thresholds, carrying
/// `label`, greyed with `why_not` when `enabled` is false.
fn unpin_entry(ui: &mut egui::Ui, label: &str, enabled: bool, why_not: &str) -> bool {
    let button = egui::Button::new(label);
    if enabled {
        ui.add(button)
            .on_hover_text(
                "Clear the verdict set by hand, and give the observation the one the \
                 thresholds propose",
            )
            .clicked()
    } else {
        ui.add_enabled(false, button)
            .on_disabled_hover_text(why_not);
        false
    }
}

/// The hover text of a greyed unpin entry on a row whose verdict is not pinned.
const NOT_PINNED: &str = "The thresholds already decide this verdict: it is not pinned.";

/// The pin in the *Keep* heading, filling `rect`: solid when any row is pinned
/// and an outline when none is, greyed when the track has no rows or the node
/// is busy. Returns whether it was clicked while it could act; the caller
/// unpins every pinned row when any is pinned, and pins every row otherwise.
fn heading_pin(
    ui: &mut egui::Ui,
    rect: egui::Rect,
    pinned: usize,
    rows: usize,
    busy: Option<&str>,
) -> bool {
    let enabled = rows > 0 && busy.is_none();
    // The drag as well, as a row's pin takes it, so a drag begun on the pin
    // does not scroll the table.
    let sense = if enabled {
        egui::Sense::click_and_drag()
    } else {
        egui::Sense::hover()
    };
    let response = ui.interact(rect, ui.id().with("track_view_heading_pin"), sense);
    response.widget_info(|| {
        egui::WidgetInfo::labeled(egui::WidgetType::Button, enabled, heading_pin_name(pinned))
    });
    if ui.is_rect_visible(rect) {
        let visuals = ui.visuals();
        let color = match (enabled, response.hovered()) {
            (true, true) => visuals.strong_text_color(),
            (true, false) => visuals.text_color(),
            (false, _) => visuals.weak_text_color().gamma_multiply(0.6),
        };
        paint_pushpin(ui.painter(), rect.center(), color, pinned > 0);
    }
    let clicked = enabled && response.clicked();
    response.on_hover_text(heading_pin_hover(pinned, rows, busy));
    clicked
}

impl TrackBody {
    /// Draw the observation table and record what it drew, with the threshold
    /// row under its headings: the boxes carry `bars_hover`, and how they moved
    /// is returned for the caller to apply.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn show_table(
        &mut self,
        ui: &mut egui::Ui,
        recon: &SfmrReconstruction,
        id: ReconId,
        state: &AppState,
        track: &EditableTrack,
        bars_hover: BoxHover<'_>,
        response: &mut TrackBodyResponse,
    ) -> BoxesMoved {
        let cols = ColumnLayout::new();
        let stage = track.stage_kind();
        let mode = self.mode().unwrap_or(BodyMode::Viewed);
        let hovered = state
            .hovered_image
            .filter(|i| i.recon == id)
            .map(ImageRef::index);
        // Only a bench track has selected rows: the viewed track has no row
        // selection, since there is no step it could be taken to.
        let selected: &[usize] = match (mode, self.showing.as_ref()) {
            (BodyMode::Edited, Some(showing)) => {
                state.selected_bench_observations(id, &showing.label)
            }
            _ => &[],
        };
        self.rows.clear();

        // The headings and the threshold row are above the scroll area, not
        // inside it: at the bottom of a long track a header that had scrolled
        // away leaves six columns of numbers with nothing saying which is
        // which. Their space is taken now and they are drawn after the rows,
        // at the horizontal offset the rows scrolled to on this frame, so a
        // sideways scroll moves them with the columns under them. Alignment
        // holds because every column is left-anchored at the table's left
        // edge, and both are placed from that edge.
        let header_band = reserve_band(ui, header_height(ui));
        let bars_band = reserve_band(ui, threshold_row_height(ui));
        let (header_rect, bars_rect) = (header_band.rect, bars_band.rect);
        let width = cols.width(mode);

        // Both ways: the table is wider than a narrow dock cell, and the
        // controls above it fit the cell's width rather than the table's. A
        // trackpad, a wheel (with Shift for sideways) and the scroll bars move
        // it, and so does a drag with the left or the middle button begun
        // anywhere on it but a control. egui drags the rows only on a touch
        // screen unless asked to always; the switches and pins take their own
        // drags, which is how a drag begun on one leaves the table where it
        // is. The headings and the threshold row are outside the scroll area,
        // so their drags are added here.
        let mut left = ui.available_rect_before_wrap().left();
        let pan = band_drag(&header_band) + band_drag(&bars_band);
        let output = egui::ScrollArea::both()
            .id_salt("track_view_rows")
            .auto_shrink([false, false])
            .scroll_source(egui::scroll_area::ScrollSource::ALL)
            .scroll_offset(self.scroll_offset - pan)
            .show(ui, |ui| {
                // The left edge the rows are drawn from on this frame. The
                // scroll area applies this frame's wheel or drag to the offset
                // it reports after drawing, so that offset is a frame ahead of
                // the rows, and the headings are placed from this edge.
                left = ui.max_rect().left();
                // Every row's Jacobian before the rows are put in order by
                // *Zoom*, whose key is read from it. Each row asks for its
                // own as it is drawn anyway, and it is geometry alone, so
                // this reads nothing that would not be read.
                if self.sort.column == SortColumn::Zoom {
                    for observation in 0..track.observations.len() {
                        self.ensure_jacobian(recon, track, observation);
                    }
                }
                for observation in self.row_order(recon, track, stage, mode) {
                    self.draw_row(
                        ui,
                        recon,
                        id,
                        state,
                        track,
                        observation,
                        stage,
                        mode,
                        hovered,
                        selected,
                        &cols,
                        response,
                    );
                }
            });
        self.scroll_offset = output.state.offset;
        let table = |band: egui::Rect| {
            egui::Rect::from_min_size(
                egui::pos2(left, band.top()),
                egui::vec2(width.max(band.width()), band.height()),
            )
        };

        let pinned: Vec<usize> = (0..track.observations.len())
            .filter(|&i| track.observations[i].pinned)
            .collect();
        let rows = track.observations.len();
        let clicked = draw_header(
            ui,
            table(header_rect),
            &cols,
            mode,
            self.sort,
            pinned.len(),
            rows,
            state.bench_edit_refusal(id).as_deref(),
        );
        if clicked.pin {
            if pinned.is_empty() {
                response.pin_verdicts = Some((0..rows).collect());
            } else {
                response.unpin_verdicts = Some(pinned);
            }
        }
        if let Some(column) = clicked.sort {
            self.sort = self.sort.clicked(column);
        }
        // Under the headings, so each bar stays beside the heading of the
        // readings it judges.
        draw_threshold_row(
            ui,
            table(bars_rect),
            bars_rect,
            &cols,
            &mut self.thresholds,
            bars_hover,
        )
    }

    /// The observations in the order the rows are drawn, by [`TrackBody::sort`].
    ///
    /// Each key is the value the row prints, so the order is the order of
    /// what a person reads: a viewed track that could not be evaluated prints
    /// no readings and no verdicts, and gives the columns that hold them no
    /// key, and the projection error is the px the cell prints.
    fn row_order(
        &self,
        recon: &SfmrReconstruction,
        track: &EditableTrack,
        stage: StageKind,
        mode: BodyMode,
    ) -> Vec<usize> {
        use crate::bench::live::Evaluation;
        let printed = !matches!(
            self.evaluation,
            Evaluation::Refused(_) | Evaluation::Failed(_)
        );
        let judged_shown = mode == BodyMode::Edited || printed;
        let keys: Vec<Option<SortKey>> = track
            .observations
            .iter()
            .enumerate()
            .map(|(i, row)| {
                let readings = |cluster: ClusterReading, tracked: TrackReading| {
                    if !printed {
                        return None;
                    }
                    match stage {
                        StageKind::Cluster => row.cluster.as_ref().and_then(cluster),
                        StageKind::Track => row.track.as_ref().and_then(tracked),
                    }
                };
                match self.sort.column {
                    SortColumn::Image => number_key(Some(f64::from(row.image))),
                    SortColumn::Verdict => self
                        .judged
                        .get(i)
                        .copied()
                        .flatten()
                        .filter(|_| judged_shown)
                        .map(|j| {
                            SortKey::Number(
                                j.checks.failed() as f64,
                                if j.proposal == Verdict::Out { 1.0 } else { 0.0 },
                            )
                        }),
                    SortColumn::Zncc => number_key(readings(|m| m.zncc, |m| m.zncc)),
                    SortColumn::SelfSimilarity => number_key(readings(
                        |m| m.zncc_self_similarity_radius,
                        |m| m.zncc_self_similarity_radius,
                    )),
                    SortColumn::ProjectionError => number_key(readings(
                        |_| None,
                        |m| m.reprojection_error.or(m.projection_offset_px),
                    )),
                    SortColumn::Shift => number_key(readings(|m| m.shift_px, |m| m.seed_shift_px)),
                    // Geometry rather than an evaluation's reading, so it is
                    // there whatever the evaluation says, as the tile is.
                    SortColumn::Zoom => number_key(
                        self.jacobians
                            .get(&i)
                            .copied()
                            .flatten()
                            .and_then(|jacobian| jacobian.mean_zoom()),
                    ),
                    SortColumn::Status => Some(SortKey::Text(
                        measurements(row, stage, &self.evaluation)[4].clone(),
                    )),
                    SortColumn::Name => Some(SortKey::Text(
                        recon
                            .image_table
                            .images
                            .get(row.image as usize)
                            .map(|im| im.name.clone())
                            .unwrap_or_else(|| format!("#{}", row.image)),
                    )),
                }
            })
            .collect();
        let images: Vec<u32> = track.observations.iter().map(|row| row.image).collect();
        sorted_rows(&keys, &images, self.sort)
    }

    /// The row's own menu, in Edited mode: what a search runs *from* is one
    /// observation, so the gesture is on the row rather than in the toolbar,
    /// exactly as the two gestures that name a pixel are in the Image Detail
    /// menu rather than here. Registered on the row's rect, so a right-click
    /// anywhere in it opens the menu for that observation.
    #[allow(clippy::too_many_arguments)]
    fn row_menu(
        &self,
        row_response: &egui::Response,
        state: &AppState,
        id: ReconId,
        track: &EditableTrack,
        observation: usize,
        stage: StageKind,
        selected: &[usize],
        response: &mut TrackBodyResponse,
    ) {
        let row = &track.observations[observation];
        let label = self
            .showing
            .as_ref()
            .map(|showing| showing.label.clone())
            .unwrap_or_default();
        // When the node's index is absent or out of date the entry itself is
        // the remedy: a person who finds the search greyed has no reason to
        // look anywhere else for it, so the row offers the build in its place.
        // The build does not then run the search -- a build over a large
        // capture takes long enough that the person has moved on, and
        // observations landing on a track unasked are a surprise.
        let sources = self.build_refusal.as_ref().and_then(|(_, why)| why.clone());
        // On a row that is one of several selected, the unpin acts on the
        // selection: every pinned verdict among the selected rows, in one step.
        let on_selection = selected.len() > 1 && selected.contains(&observation);
        let (unpin_label, unpin_rows, why_not) = if on_selection {
            let rows: Vec<usize> = selected
                .iter()
                .copied()
                .filter(|&i| track.observations.get(i).is_some_and(|o| o.pinned))
                .collect();
            (
                unpin_selection_label(rows.len()),
                rows,
                "None of the selected verdicts is pinned: the thresholds already decide them.",
            )
        } else {
            let rows = if row.pinned {
                vec![observation]
            } else {
                Vec::new()
            };
            (UNPIN_LABEL.to_string(), rows, NOT_PINNED)
        };
        // A view-only bench or a busy node greys the unpin whatever the rows.
        let refusal = state.bench_edit_refusal(id);
        let (unpin_rows, why_not) = match &refusal {
            Some(why) => (Vec::new(), why.as_str()),
            None => (unpin_rows, why_not),
        };
        crate::context_menu::on_secondary_click(row_response).show(|ui| {
            if unpin_entry(ui, &unpin_label, !unpin_rows.is_empty(), why_not) {
                response.unpin_verdicts = Some(unpin_rows.clone());
                ui.close();
            }
            ui.separator();
            match state.sift_index_state(id) {
                crate::index_files::IndexFileState::Current => {
                    let button = egui::Button::new(super::SEARCH_DESCRIPTORS_LABEL);
                    let clicked = match state.bench_search_refusal(id, &label, observation) {
                        None => ui
                            .add(button)
                            .on_hover_text(
                                "Ask the SIFT index which other photographs hold the patch \
                                 around this observation, and add each to the track",
                            )
                            .clicked(),
                        Some(why) => {
                            ui.add_enabled(false, button).on_disabled_hover_text(why);
                            false
                        }
                    };
                    if clicked {
                        response.search_descriptors = Some(observation);
                        ui.close();
                    }
                }
                _ => {
                    let text = state.index_files_build_label(id);
                    // The staleness sentence where there is one, so the entry
                    // says what is wrong as well as what to do about it.
                    let hint = state
                        .sift_index(id)
                        .and_then(|index| index.stale_reason())
                        .map(str::to_string)
                        .unwrap_or_else(|| {
                            "No SIFT index is open. Build this reconstruction's index files, \
                             its SIFT index and its cluster patches, beside its .sfmr. Runs on \
                             a worker thread, and does not then run the search."
                                .to_string()
                        });
                    let refusal = state
                        .busy_refusal(id)
                        .or_else(|| state.index_files_home_refusal(id))
                        .or(sources.clone());
                    let button = egui::Button::new(text);
                    let clicked = match refusal {
                        None => ui.add(button).on_hover_text(hint).clicked(),
                        Some(why) => {
                            ui.add_enabled(false, button).on_disabled_hover_text(why);
                            false
                        }
                    };
                    if clicked {
                        response.build_index_files = true;
                        ui.close();
                    }
                }
            }
            if stage == StageKind::Track {
                let button = egui::Button::new(super::SEARCH_GEOMETRY_LABEL);
                let clicked = match state.bench_geometry_search_refusal(id, &label, observation) {
                    None => ui
                        .add(button)
                        .on_hover_text(
                            "Project this track's patch into the reconstruction's other cameras, \
                             vet their patches against this observation and the accepted views, \
                             and add each match to the track",
                        )
                        .clicked(),
                    Some(why) => {
                        ui.add_enabled(false, button).on_disabled_hover_text(why);
                        false
                    }
                };
                if clicked {
                    response.search_geometry = Some(observation);
                    ui.close();
                }
            }
            // A sighting the last fit kept at its seed can be taken where the
            // walk would have put it. Offered on those rows alone: the entry
            // is the overruling of one refusal, and a row with no refusal has
            // nothing to overrule.
            let walk = (stage == StageKind::Track)
                .then(|| accepted_walk(row))
                .flatten();
            if let Some(walk) = walk {
                let button = egui::Button::new(super::ACCEPT_WALK_LABEL);
                let clicked = match state.bench_edit_refusal(id) {
                    None => ui.add(button).on_hover_text(walk).clicked(),
                    Some(why) => {
                        ui.add_enabled(false, button).on_disabled_hover_text(why);
                        false
                    }
                };
                if clicked {
                    response.accept_walk = Some(observation);
                    ui.close();
                }
            }
        });
    }

    /// A row's *Keep* switch and its pin, in Edited mode, registered after the
    /// row so they take the clicks that land on them, each over the whole
    /// height of the row: a two-state decision is one switch, and a cell-sized
    /// target is easy to hit. Returns the switch's hover text.
    ///
    /// With a `refusal` (the node busy, or a view-only bench) both are drawn
    /// as they stand but greyed, do nothing with a click, and carry the
    /// refusal as their hover.
    #[allow(clippy::too_many_arguments)]
    fn draw_keep(
        &self,
        ui: &mut egui::Ui,
        rect: egui::Rect,
        cols: &ColumnLayout,
        row: &sfmtool_core::bench::Observation,
        observation: usize,
        judged: Option<&Judgement>,
        refusal: Option<&str>,
        response: &mut TrackBodyResponse,
    ) -> String {
        let x0 = rect.min.x;
        let keep_rect = egui::Rect::from_min_size(
            egui::pos2(x0 + cols.keep, rect.min.y),
            egui::vec2(SWITCH_CELL_WIDTH, ROW_HEIGHT),
        );
        let pin_rect = egui::Rect::from_min_max(
            egui::pos2(keep_rect.right(), rect.min.y),
            egui::pos2(x0 + cols.keep + KEEP_WIDTH, rect.max.y),
        );
        let kept = row.verdict == Verdict::In;
        let enabled = refusal.is_none();
        let keep = keep_switch(
            ui,
            keep_rect,
            ui.id().with(("track_view_keep", observation)),
            kept,
            enabled,
        );
        let pin = pin_toggle(
            ui,
            pin_rect,
            ui.id().with(("track_view_pin", observation)),
            row.pinned,
            enabled,
        );
        let hover = keep_hover(kept, row.pinned, judged, row.image);
        if let Some(why) = refusal {
            pin.on_hover_text(why);
            keep.on_hover_text(why);
            return hover;
        }
        // Pinning the verdict a row already carries is `set_verdict` with that
        // verdict, and unpinning is `unpin_verdicts` of that row.
        if pin.clicked() {
            if row.pinned {
                response.unpin_verdicts = Some(vec![observation]);
            } else {
                response.set_verdict = Some((observation, row.verdict));
            }
        }
        pin.on_hover_text(pin_hover(row.pinned));
        if keep.clicked() {
            response.set_verdict =
                Some((observation, if kept { Verdict::Out } else { Verdict::In }));
        }
        crate::context_menu::on_secondary_click(&keep).show(|ui| {
            if unpin_entry(ui, UNPIN_LABEL, row.pinned, NOT_PINNED) {
                response.unpin_verdicts = Some(vec![observation]);
                ui.close();
            }
        });
        keep.on_hover_text(&hover);
        hover
    }

    /// One observation: its painting, its verdict control, its tile and its
    /// cells.
    #[allow(clippy::too_many_arguments)]
    fn draw_row(
        &mut self,
        ui: &mut egui::Ui,
        recon: &SfmrReconstruction,
        id: ReconId,
        state: &AppState,
        track: &EditableTrack,
        observation: usize,
        stage: StageKind,
        mode: BodyMode,
        hovered: Option<usize>,
        selected: &[usize],
        cols: &ColumnLayout,
        response: &mut TrackBodyResponse,
    ) {
        let edited = mode == BodyMode::Edited;
        let row = &track.observations[observation];
        let image = ImageRef::new(id, row.image as usize);
        let judged = self.judged.get(observation).copied().flatten();
        // A viewed track that cannot be evaluated prints no reading, and its
        // *Verdict* cells say nothing either: what the bars would give a row
        // is a statement about its readings.
        let refused = matches!(
            self.evaluation,
            crate::bench::live::Evaluation::Refused(_) | crate::bench::live::Evaluation::Failed(_)
        );
        let judged = judged.filter(|_| edited || !refused);
        let proposal = judged.map(|j| j.proposal);
        let cells = measurements(row, stage, &self.evaluation);
        let checks = cell_checks(judged.as_ref(), &self.evaluation);
        let grids = row_grids(row, stage, &self.evaluation);
        let name = recon
            .image_table
            .images
            .get(image.index())
            .map(|im| im.name.clone())
            .unwrap_or_else(|| format!("#{}", row.image));

        // As wide as the table, or the panel when it is wider, so the scroll
        // area knows how far there is to scroll sideways.
        let available = ui.available_rect_before_wrap();
        let width = available.width().max(cols.width(mode));
        let rect = egui::Rect::from_min_size(available.min, egui::vec2(width, ROW_HEIGHT));
        let row_response = ui.allocate_rect(rect, egui::Sense::click());

        // The verdict cell's tint first, then the selection and the hover
        // over the whole row: what the bars propose is a property of the
        // numbers, and which row the pointer or the split is on is a property
        // of this frame. Only the verdict cell is tinted, so a *Keep* switch
        // in it reads as the person's decision set against the bars' proposal.
        let visuals = ui.visuals();
        let keep_cell = egui::Rect::from_min_max(
            egui::pos2(rect.min.x + cols.keep, rect.min.y),
            egui::pos2(rect.min.x + cols.keep + KEEP_WIDTH, rect.max.y),
        );
        let tint = proposal_tint(visuals, proposal);
        if let Some(tint) = tint {
            ui.painter().rect_filled(keep_cell, 0.0, tint);
        }
        if selected.contains(&observation) {
            ui.painter()
                .rect_filled(rect, 0.0, visuals.selection.bg_fill.gamma_multiply(0.45));
        }
        if row_response.hovered() || hovered == Some(image.index()) {
            ui.painter().rect_filled(
                rect,
                0.0,
                visuals.widgets.hovered.bg_fill.gamma_multiply(0.4),
            );
        }
        if row_response.hovered() {
            response.hovered_image = Some(image.index());
        }
        // The row's own menu, in Edited mode only: every entry in it is a
        // step on the track.
        if edited {
            self.row_menu(
                &row_response,
                state,
                id,
                track,
                observation,
                stage,
                selected,
                response,
            );
        }

        if row_response.clicked() {
            response.select_image = Some(image.index());
            // With the image, where in it this observation sits, so the Image
            // Detail panel can bring it into view: the same position its bench
            // layer draws the mark at.
            response.reveal_feature = crate::bench::observation_pixel(row);
            // Only a bench track has a row selection, which *Split* reads.
            if edited {
                let extend = ui.input(|i| i.modifiers.command || i.modifiers.shift);
                response.pick_row = Some((observation, extend));
            }
        }
        // Camera view for the row's image, in either mode. The observation
        // goes with it, so the view turns to show it.
        if row_response.double_clicked() {
            response.request_camera_view = Some(image.index());
            response.reveal_feature = crate::bench::observation_pixel(row);
        }

        let x0 = rect.min.x;
        let cy = rect.center().y;

        let (keep_hover, verdict_text) = if edited {
            (
                self.draw_keep(
                    ui,
                    rect,
                    cols,
                    row,
                    observation,
                    judged.as_ref(),
                    state.bench_edit_refusal(id).as_deref(),
                    response,
                ),
                None,
            )
        } else {
            let (hover, text) =
                draw_verdict(ui, rect, cols, observation, judged.as_ref(), row.image);
            (hover, Some(text))
        };

        // The tile: rendered once per observation and kept until the track or
        // the item moves, because a warp per row per frame is a warp per row
        // per frame. The rect is drawn whether or not there is a tile in it, so
        // the columns beside it do not shift when one cannot be rendered.
        let tile_rect = egui::Rect::from_min_size(
            egui::pos2(x0 + cols.tile, cy - TILE_SIZE / 2.0),
            egui::vec2(TILE_SIZE, TILE_SIZE),
        );
        let tile = self.ensure_tile(ui.ctx(), recon, track, observation, state);
        match tile {
            Some(texture) => {
                ui.painter().image(
                    texture,
                    tile_rect,
                    egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                    egui::Color32::WHITE,
                );
            }
            None => {
                ui.painter()
                    .rect_filled(tile_rect, 2.0, ui.visuals().faint_bg_color);
            }
        }
        // Hovering the tile shows it in context: the same picture over a wider
        // stretch of the photograph, with the patch boxed in it and the
        // projection the *Proj. err* cell measures to. A hover sense takes no
        // click, so a click on the tile is still the row's.
        if tile.is_some() {
            ui.interact(
                tile_rect,
                ui.id().with(("track_view_tile", observation)),
                egui::Sense::hover(),
            )
            .on_hover_ui(|ui| {
                ui.label(egui::RichText::new(&name).strong());
                match self.ensure_context(ui.ctx(), recon, track, observation, state) {
                    Some(drawn) => drawn.show(ui),
                    None => {
                        ui.label("Nothing around this patch could be rendered.");
                    }
                }
            });
        }

        // The crop: the photograph around the patch's outline as it lands
        // there, fitted to a tile-sized cell. Its hover view is the crop in
        // three times its width and height of the photograph, with the
        // patch's axes in the photograph's pixels.
        let crop_rect = egui::Rect::from_min_size(
            egui::pos2(x0 + cols.crop, cy - TILE_SIZE / 2.0),
            egui::vec2(TILE_SIZE, TILE_SIZE),
        );
        ui.painter()
            .rect_filled(crop_rect, 2.0, ui.visuals().faint_bg_color);
        let cropped = match self.ensure_crop(ui.ctx(), recon, track, observation, state) {
            Some(drawn) => {
                drawn.paint_cell(ui.painter(), crop_rect);
                true
            }
            None => false,
        };
        // The pixel and the feature index, the last bullets under the hover
        // view's own.
        let crop_caption = if cropped {
            let origin = self.showing.as_ref().and_then(|showing| showing.origin);
            crate::scene::node_by_id(&state.scene, id)
                .and_then(|node| super::crop_caption(node.edited(), origin, row))
        } else {
            None
        };
        if cropped {
            ui.interact(
                crop_rect,
                ui.id().with(("track_view_crop", observation)),
                egui::Sense::hover(),
            )
            .on_hover_ui(|ui| {
                ui.label(egui::RichText::new(&name).strong());
                match self.ensure_crop_context(ui.ctx(), recon, track, observation, state) {
                    Some(drawn) => drawn.show(ui),
                    None => {
                        ui.label("Nothing around this crop could be rendered.");
                    }
                }
                if let Some(caption) = &crop_caption {
                    ui.label(caption);
                }
            });
        }

        let painter = ui.painter();
        let font = egui::TextStyle::Body.resolve(ui.style());
        let text_color = ui.visuals().text_color();
        let weak = ui.visuals().weak_text_color();
        let text = |x: f32, value: &str, color: egui::Color32| {
            painter.text(
                egui::pos2(x0 + x, cy),
                egui::Align2::LEFT_CENTER,
                value,
                font.clone(),
                color,
            );
        };
        text(cols.image, &format!("{}", row.image), text_color);
        let name_x = cols.name_x(mode);
        let shown = crate::elide::middle(&name, NAME_WIDTH - 8.0, |value| {
            ui.ctx().fonts_mut(|fonts| {
                fonts
                    .layout_no_wrap(value.to_owned(), font.clone(), weak)
                    .rect
                    .width()
            })
        });
        text(name_x, &shown, weak);
        // The name is elided in its middle to fit the column, so hovering the
        // column shows it whole. Hovering the image's index shows the same
        // name, since the index is at the table's left edge and the name at
        // its right.
        for (which, from, to) in [
            ("track_view_name", name_x, name_x + NAME_WIDTH - 8.0),
            ("track_view_image", cols.image, cols.crop - 4.0),
        ] {
            let cell = egui::Rect::from_min_max(
                egui::pos2(x0 + from, rect.min.y),
                egui::pos2(x0 + to, rect.max.y),
            );
            ui.interact(
                cell,
                ui.id().with((which, observation)),
                egui::Sense::hover(),
            )
            .on_hover_text(&name);
        }
        // While an evaluation of new inputs is on its way the numbers are the
        // last evaluation's, and they are greyed so that they do not read as
        // the numbers of the track as it now stands.
        let number_color = match self.evaluation {
            crate::bench::live::Evaluation::Evaluating => weak,
            _ => text_color,
        };
        // The colours fade with the numbers, as the grids do, so a stale
        // number does not read as judged.
        let fade = if number_color == weak { 0.45 } else { 1.0 };
        // Each line of a cell in the colour its own bar gives it, laid out as
        // one galley so the two lines stack exactly as one string would.
        let lines = |x: f32, value: &str, colors: [egui::Color32; 2]| {
            let mut job = egui::text::LayoutJob::default();
            for (k, line) in value.split('\n').enumerate() {
                let line = if k == 0 {
                    line.to_string()
                } else {
                    format!("\n{line}")
                };
                job.append(
                    &line,
                    0.0,
                    egui::TextFormat::simple(font.clone(), colors[k.min(1)]),
                );
            }
            let galley = painter.layout_job(job);
            let at = egui::Align2::LEFT_CENTER
                .anchor_size(egui::pos2(x0 + x, cy), galley.size())
                .min;
            painter.galley(at, galley, colors[0]);
        };
        for ((x, cell), check) in [cols.zncc, cols.shift, cols.offset, cols.self_similarity]
            .into_iter()
            .zip(cells.iter())
            .zip(checks.iter())
        {
            let colors = check.map(|check| match check {
                BarCheck::NotJudged => number_color,
                judged => check_color(ui.visuals(), judged, number_color).gamma_multiply(fade),
            });
            lines(x, cell, colors);
        }
        // Hovering the self-similarity numbers shows the ellipses they are
        // the semi-major axes of, in grid px, image px and along the patch's
        // axes. The table is built only while the cell is hovered.
        let self_similarity_ellipse = row_ellipse(row, stage, &self.evaluation);
        if self_similarity_ellipse.iter().any(Option::is_some) {
            let cell = egui::Rect::from_min_max(
                egui::pos2(x0 + cols.self_similarity, rect.min.y),
                egui::pos2(x0 + cols.self_similarity_grid - 4.0, rect.max.y),
            );
            let world_unit = recon.metadata.world_space_unit.as_deref();
            ui.interact(
                cell,
                ui.id().with(("track_view_self_similarity", observation)),
                egui::Sense::hover(),
            )
            .on_hover_ui(|ui| {
                let [whole, middle] = &self_similarity_ellipse;
                if let Some(text) =
                    self_similarity_ellipse_text(whole.as_ref(), middle.as_ref(), world_unit)
                {
                    ui.label(SELF_SIMILARITY_ELLIPSE_CAPTION);
                    ui.label(egui::RichText::new(text).monospace());
                }
            });
        }
        // The status cell is a sentence rather than a number at the track
        // stage, so it is elided to its column the way the image name is.
        let status = crate::elide::middle(&cells[4], cols.from - cols.status - 8.0, |value| {
            ui.ctx().fonts_mut(|fonts| {
                fonts
                    .layout_no_wrap(value.to_owned(), font.clone(), text_color)
                    .rect
                    .width()
            })
        });
        text(cols.status, &status, text_color);
        // The tile's magnification is geometry, not a reading of the
        // evaluation, so it is in the plain text colour whatever the
        // evaluation stands at, as the tile is.
        let jacobian = self.ensure_jacobian(recon, track, observation);
        let zoom_text = zoom_text(jacobian);
        text(cols.zoom, &zoom_text, text_color);
        // Hovering the zoom says which sampler the tile is rendered with,
        // since the sampler rule reads it from the same Jacobian.
        if let Some(jacobian) = jacobian {
            let cell = egui::Rect::from_min_max(
                egui::pos2(x0 + cols.zoom, rect.min.y),
                egui::pos2(x0 + cols.status - 4.0, rect.max.y),
            );
            ui.interact(
                cell,
                ui.id().with(("track_view_zoom", observation)),
                egui::Sense::hover(),
            )
            .on_hover_text(zoom_sampler_text(&jacobian));
        }
        if edited {
            text(cols.from, &provenance_text(row.provenance), weak);
        }

        // The two grids, faded with the numbers while an evaluation is on its
        // way. Each is laid out as the tile is, so a cell sits over the part
        // of the tile it read.
        let drawn = [
            (
                cols.zncc_grid,
                grids.zncc,
                None,
                &zncc_cell_color as &dyn Fn(f64) -> Option<egui::Color32>,
                GridKind::Zncc,
            ),
            (
                cols.self_similarity_grid,
                grids.radius,
                grids
                    .radius_ellipse
                    .map(|ellipses| ellipses.map(|row| row.map(|e| ellipse_mark(&e)))),
                &self_similarity_cell_color,
                GridKind::SelfSimilarity,
            ),
        ];
        for (x, grid, marks, color, which) in drawn {
            let Some(grid) = grid else {
                continue;
            };
            let grid_rect = egui::Rect::from_min_size(
                egui::pos2(x0 + x, cy - GRID_SIDE / 2.0),
                egui::vec2(GRID_SIDE, GRID_SIDE),
            );
            draw_grid(ui, grid_rect, &grid, marks.as_ref(), color, fade);
            ui.interact(
                grid_rect,
                ui.id().with(("track_view_grid", observation, which)),
                egui::Sense::hover(),
            )
            .on_hover_ui(|ui| {
                ui.label(egui::RichText::new(grid_numbers(&grid, which)).monospace());
            });
        }

        // The whole tile's self-similarity surface, with the contour round the
        // region its radius is read from, and a larger one on hover.
        let mut plotted = false;
        if let Some((surface, tolerance)) = row_surface(row, stage, &self.evaluation) {
            let radius = row_radius(row, stage);
            if let Some(drawn) = self.ensure_plot(ui.ctx(), observation, surface, tolerance) {
                plotted = true;
                let plot_rect = egui::Rect::from_min_size(
                    egui::pos2(x0 + cols.self_similarity_plot, cy - PLOT_SIDE / 2.0),
                    egui::vec2(PLOT_SIDE, PLOT_SIDE),
                );
                drawn.paint(ui.painter(), plot_rect, fade);
                ui.interact(
                    plot_rect,
                    ui.id().with(("track_view_plot", observation)),
                    egui::Sense::hover(),
                )
                .on_hover_ui(|ui| {
                    let (rect, _) = ui.allocate_exact_size(
                        egui::vec2(PLOT_HOVER_SIDE, PLOT_HOVER_SIDE),
                        egui::Sense::hover(),
                    );
                    drawn.paint(ui.painter(), rect, 1.0);
                    ui.label(plot_caption(radius, tolerance));
                });
            }
        }

        self.rows.push(RowSummary {
            observation,
            image: row.image,
            verdict: row.verdict,
            pinned: row.pinned,
            proposal,
            keep_hover,
            verdict_text,
            tint,
            crop_caption,
            cells,
            self_similarity_ellipse,
            jacobian,
            zoom_text,
            checks,
            grids,
            tile: tile.is_some(),
            crop: cropped,
            self_similarity_plot: plotted,
        });
    }
}

/// The line over the *Self-similarity* cell's hover table.
pub(super) const SELF_SIMILARITY_ELLIPSE_CAPTION: &str = "Where a match could land and still \
    look like the true position, for the whole patch and its middle: the ellipse with the same \
    spread about the true position as the shifts that match, as its semi-major axis \
    × its semi-minor axis and the angle of the major axis. In patch-grid px, the semi-major \
    axis is the radius, and the angle runs from grid x (right) towards grid y (down); in the \
    photograph's pixels, from its x towards its y (down); along the patch, from u towards v. \
    A + is a lower bound, so the true length may be larger: the match ran off the search \
    square, met a shift with no reading, or reached the largest radius searched.";

/// The words under the hover view of a surface plot: the radius and the
/// level the contour is drawn at.
pub(super) fn plot_caption(radius: Option<f64>, tolerance: f64) -> String {
    let radius = radius.map_or_else(|| "-".to_string(), super::radius_number);
    format!(
        "ZNCC of the patch against itself at every shift up to 3 px along each axis. The \
         contour is at {:.3}, one minus the tolerance {:.3}; the dots are the whole-pixel \
         shifts inside it. Radius {radius}: the semi-major axis of the ellipse with the \
         same spread about the centre as the region inside the contour, read up to 3 px.",
        1.0 - tolerance,
        tolerance
    )
}

/// Paint a three-by-three grid into `rect`: a border, and each cell filled
/// in the colour `color` gives its value, or the faint background where it
/// gives none. With `marks`, each cell also carries its [`CellMark`], drawn
/// dark over the colour.
fn draw_grid(
    ui: &egui::Ui,
    rect: egui::Rect,
    grid: &[[f64; 3]; 3],
    marks: Option<&[[CellMark; 3]; 3]>,
    color: &dyn Fn(f64) -> Option<egui::Color32>,
    fade: f32,
) {
    let painter = ui.painter();
    let visuals = ui.visuals();
    painter.rect_filled(rect, 0.0, visuals.weak_text_color());
    for (row, values) in grid.iter().enumerate() {
        for (col, &value) in values.iter().enumerate() {
            let min = rect.min
                + egui::vec2(
                    1.0 + col as f32 * (GRID_CELL + 1.0),
                    1.0 + row as f32 * (GRID_CELL + 1.0),
                );
            let cell = egui::Rect::from_min_size(min, egui::vec2(GRID_CELL, GRID_CELL));
            let fill = color(value).unwrap_or(visuals.faint_bg_color);
            painter.rect_filled(cell, 0.0, fill.gamma_multiply(fade));
            let stroke = egui::Stroke::new(
                1.5,
                egui::Color32::from_black_alpha(220).gamma_multiply(fade),
            );
            let c = cell.center();
            match marks.map_or(CellMark::Nothing, |m| m[row][col]) {
                CellMark::Nothing => {}
                CellMark::Line(half) => {
                    painter.line_segment([c - half, c + half], stroke);
                }
            }
        }
    }
}

/// What a self-similarity grid cell draws over its colour: the direction that
/// ninth of the tile can slide in, along its ellipse's major axis, or nothing.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) enum CellMark {
    /// No direction stands out; the colour says the rest.
    Nothing,
    /// A match could slide along the line: the half line from the cell's
    /// centre, in screen points. The grid frame's `x` is the screen's right
    /// and its `y` the screen's down, so the major axis is drawn as it is.
    Line(egui::Vec2),
}

/// The [`CellMark`] of a self-similarity cell with ellipse `ellipse`: a line
/// along its major axis where the ellipse is long and thin, its elongation
/// `1 − (minor / major)²` at least `0.5` (a minor axis at most 0.71 of the
/// major), so the cell's matching shifts line up along one direction, and
/// nothing otherwise, nor for a circle or a cell with no reading.
pub(super) fn ellipse_mark(ellipse: &SelfSimilarityEllipse) -> CellMark {
    let [major, minor] = ellipse.axes;
    let angle = ellipse.major_angle;
    if !(angle.is_finite() && major > 0.0) {
        return CellMark::Nothing;
    }
    let elongation = 1.0 - (minor / major).powi(2);
    if elongation.is_nan() || elongation < 0.5 {
        return CellMark::Nothing;
    }
    let (sin, cos) = angle.sin_cos();
    CellMark::Line(egui::vec2(cos as f32, sin as f32) * (GRID_CELL / 2.0 - 1.0))
}

/// Which of a row's grids a value belongs to, which says how its hover text
/// prints it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) enum GridKind {
    /// A ZNCC, printed in percent.
    Zncc,
    /// A self-similarity radius, printed as the cell prints it.
    SelfSimilarity,
}

/// A grid's nine values as its hover text shows them, three to a line: a
/// ZNCC in percent, a self-similarity radius as its cell prints it (`1.4`, `3+`), and `-` for a cell with no reading.
pub(super) fn grid_numbers(grid: &[[f64; 3]; 3], kind: GridKind) -> String {
    grid.iter()
        .map(|row| {
            row.iter()
                .map(|&value| match (value.is_finite(), kind) {
                    (false, _) => format!("{:>5}", "-"),
                    (true, GridKind::Zncc) => format!("{:>5.0}", 100.0 * value),
                    (true, GridKind::SelfSimilarity) => {
                        format!("{:>5}", radius_number(value))
                    }
                })
                .collect::<Vec<_>>()
                .join(" ")
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// What *Accept walk* would do to `row`, as its hover text, or `None` for a row
/// the last fit did not keep at its seed.
///
/// The numbers a person decides by: how far, to where, and the ZNCC at each
/// end -- the row's own, which the reading after the fit took at the seed, and
/// the one the fit's localizer scored at the walked peak.
pub(super) fn accepted_walk(row: &sfmtool_core::bench::Observation) -> Option<String> {
    let m = row.track.as_ref()?;
    let to = m.walked_to?;
    let zncc = |value: Option<f64>, middle: Option<f64>| match value {
        Some(v) if v.is_finite() => super::zncc_sentence(Some(v), middle),
        _ => "not scored".to_string(),
    };
    Some(format!(
        "Move this sighting {:.1} grid px, to ({:.1}, {:.1}), where the last fit's walk \
         would have put it. ZNCC {} at the seed, {} at the walked peak. Pins it, \
         as a hand placement does.",
        m.walked_px.unwrap_or(f64::NAN),
        to[0],
        to[1],
        zncc(m.zncc, m.zncc_middle),
        zncc(m.walked_zncc, m.walked_zncc_middle),
    ))
}

/// Take a band `height` tall across the whole width `ui` offers, for something
/// drawn into it later in the frame. The band senses a drag, which scrolls
/// the table; a control drawn into it later is on top and takes the pointer
/// over itself.
fn reserve_band(ui: &mut egui::Ui, height: f32) -> egui::Response {
    let available = ui.available_rect_before_wrap();
    let rect = egui::Rect::from_min_size(available.min, egui::vec2(available.width(), height));
    ui.allocate_rect(rect, egui::Sense::drag())
}

/// How far a drag of `band` with the left or the middle button moved the
/// pointer on this frame: the buttons the rows' scroll area drags by.
fn band_drag(band: &egui::Response) -> egui::Vec2 {
    if band.dragged_by(egui::PointerButton::Primary) || band.dragged_by(egui::PointerButton::Middle)
    {
        band.drag_delta()
    } else {
        egui::Vec2::ZERO
    }
}

/// The headings' height: body-sized, as the cells under them are, with room
/// around them.
fn header_height(ui: &egui::Ui) -> f32 {
    ui.text_style_height(&egui::TextStyle::Body) + 8.0
}

/// The threshold row's height: two lines of boxes, for the ZNCC column's
/// whole over mid.
fn threshold_row_height(ui: &egui::Ui) -> f32 {
    2.0 * ui.spacing().interact_size.y + ui.spacing().item_spacing.y + 4.0
}

/// The header row, painted in `rect`, whose left edge is the table's, at the
/// same offsets the rows draw at: above the scroll area, so it stays put while
/// the rows move up and down under it, and moved sideways with them. In the
/// weak colour, so the headings still read as headings.
///
/// In Edited mode the *Keep* heading carries a pin over the rows' pin column,
/// which unpins every pinned verdict of the track, or pins every verdict as it
/// stands when none is pinned; `pinned` is how many are pinned, `rows` how
/// many observations the track has and `busy` the node's busy refusal.
///
/// A heading that orders the rows takes a click, and the one `sort` orders
/// them by carries a triangle after its word, pointing up for increasing and
/// down for decreasing. Returns what was clicked.
#[allow(clippy::too_many_arguments)]
fn draw_header(
    ui: &mut egui::Ui,
    rect: egui::Rect,
    cols: &ColumnLayout,
    mode: BodyMode,
    sort: TableSort,
    pinned: usize,
    rows: usize,
    busy: Option<&str>,
) -> HeaderClicks {
    let font = egui::TextStyle::Body.resolve(ui.style());
    let headers = cols.headers(mode);
    let mut clicks = HeaderClicks::default();
    for (k, &(x, label, tip)) in headers.iter().enumerate() {
        // The heading's region runs to where the next one starts, so the
        // whole width of its column answers, not only the word.
        let end = headers
            .get(k + 1)
            .map_or(rect.max.x, |&(next, _, _)| rect.min.x + next);
        let cell = egui::Rect::from_x_y_ranges(rect.min.x + x..=end, rect.y_range());
        let column = SortColumn::of_heading(label);
        let sense = if column.is_some() {
            egui::Sense::click()
        } else {
            egui::Sense::hover()
        };
        let response = ui.interact(cell, ui.id().with(("track_view_heading", k)), sense);
        let color = if column.is_some() && response.hovered() {
            ui.visuals().text_color()
        } else {
            ui.visuals().weak_text_color()
        };
        let word = ui.painter().text(
            egui::pos2(rect.min.x + x, rect.center().y),
            egui::Align2::LEFT_CENTER,
            label,
            font.clone(),
            color,
        );
        if let Some(column) = column {
            let ordered = sort.column == column;
            if ordered {
                paint_sort_triangle(
                    ui.painter(),
                    egui::pos2(
                        word.right() + 2.0 + SORT_TRIANGLE_HALF_WIDTH,
                        rect.center().y,
                    ),
                    sort.descending,
                    color,
                );
            }
            response.widget_info(|| {
                egui::WidgetInfo::labeled(
                    egui::WidgetType::Button,
                    true,
                    format!("Sort by {}", column.name()),
                )
            });
            if response.clicked() {
                clicks.sort = Some(column);
            }
            response.on_hover_text(format!("{tip}\n\n{}", sort_hover(sort, column)));
        } else {
            response.on_hover_text(tip);
        }
    }
    if mode == BodyMode::Edited {
        // After the headings, so it takes the pointer over its own few points.
        let pin_rect = egui::Rect::from_min_max(
            egui::pos2(rect.min.x + cols.keep + SWITCH_CELL_WIDTH, rect.min.y),
            egui::pos2(rect.min.x + cols.keep + KEEP_WIDTH, rect.max.y),
        );
        clicks.pin = heading_pin(ui, pin_rect, pinned, rows, busy);
    }
    clicks
}

/// What a click on the header asked for.
#[derive(Debug, Default)]
struct HeaderClicks {
    /// The *Keep* heading's pin was clicked.
    pin: bool,
    /// A heading was clicked, which orders the rows by its column.
    sort: Option<SortColumn>,
}

/// The last paragraph of a heading's hover text that orders the rows: what a
/// click on it does with the order `sort` stands at.
pub(super) fn sort_hover(sort: TableSort, column: SortColumn) -> &'static str {
    match (sort.column == column, sort.descending) {
        (true, false) => "The rows are in increasing order by this column. Click to reverse it.",
        (true, true) => "The rows are in decreasing order by this column. Click to reverse it.",
        (false, _) if column.worst_first_descending() => {
            "Click to put the rows in decreasing order by this column, worst first."
        }
        (false, _) if column == SortColumn::Zncc => {
            "Click to put the rows in increasing order by this column, worst first."
        }
        (false, _) => "Click to put the rows in increasing order by this column.",
    }
}

/// The small triangle beside the heading the rows are ordered by, centred at
/// `c`: pointing up for increasing order and down for decreasing.
fn paint_sort_triangle(
    painter: &egui::Painter,
    c: egui::Pos2,
    descending: bool,
    color: egui::Color32,
) {
    let (half, rise) = (SORT_TRIANGLE_HALF_WIDTH, 3.0);
    // Clockwise on the screen, as egui's tessellator wants a filled shape.
    let points = if descending {
        vec![
            egui::pos2(c.x - half, c.y - rise),
            egui::pos2(c.x + half, c.y - rise),
            egui::pos2(c.x, c.y + rise),
        ]
    } else {
        vec![
            egui::pos2(c.x - half, c.y + rise),
            egui::pos2(c.x, c.y - rise),
            egui::pos2(c.x + half, c.y + rise),
        ]
    };
    painter.add(egui::Shape::convex_polygon(
        points,
        color,
        egui::Stroke::NONE,
    ));
}

/// The threshold row under the headings: each bar's box in the column whose
/// readings it judges, followed by the unit and the name those readings print
/// with, so `[70]% whole` stands over `93% whole`. The two ZNCC bars stack as
/// the ZNCC cell stacks its two readings. The columns no bar judges are empty,
/// so the word *Thresholds* takes the left of the row. Drawn the same in both
/// modes, and greyed when `hover` is a refusal.
fn draw_threshold_row(
    ui: &mut egui::Ui,
    rect: egui::Rect,
    band: egui::Rect,
    cols: &ColumnLayout,
    bars: &mut Thresholds,
    hover: BoxHover<'_>,
) -> BoxesMoved {
    let line = ui.spacing().interact_size.y;
    let gap = ui.spacing().item_spacing.y;
    // A box scrolled out of the band it stands in is clipped to it, as a row
    // is to the scroll area.
    let clip = ui.clip_rect().intersect(band);
    ui.painter().text(
        egui::pos2(rect.min.x + cols.image, rect.min.y + 0.5 * line),
        egui::Align2::LEFT_CENTER,
        "Thresholds",
        egui::TextStyle::Body.resolve(ui.style()),
        ui.visuals().weak_text_color(),
    );
    // Each box with the column it sits in, the line of that column's cell it
    // stands beside, the text after it and that text's hover. The ZNCC bars
    // read in percent, as the ZNCC cell does; the track stores them on the 0
    // to 1 scale. Listed left to right and top to bottom, which is the order
    // Tab moves through them.
    let boxes = [
        (
            cols.zncc,
            0,
            percent(egui::DragValue::new(&mut bars.min_zncc)),
            MIN_ZNCC_LABEL,
            MIN_ZNCC_TIP,
        ),
        // The bar on the middle reading; 0 turns it off.
        (
            cols.zncc,
            1,
            percent(egui::DragValue::new(&mut bars.min_zncc_middle)),
            MIN_ZNCC_MIDDLE_LABEL,
            MIN_ZNCC_MIDDLE_TIP,
        ),
        // In patch-grid px, the unit of the self-similarity column; at the
        // largest radius read it turns nothing out.
        (
            cols.self_similarity,
            0,
            egui::DragValue::new(&mut bars.max_zncc_self_similarity_radius)
                .range(0.0..=max_self_similarity_radius())
                .speed(0.02)
                .max_decimals(1),
            MAX_SELF_SIMILARITY_LABEL,
            MAX_SELF_SIMILARITY_TIP,
        ),
        // In patch-grid px, and also the radius the evaluation looks for each
        // peak within.
        (
            cols.shift,
            0,
            egui::DragValue::new(&mut bars.max_shift_px)
                .range(0.0..=24.0)
                .speed(0.05)
                .max_decimals(1),
            MAX_SHIFT_LABEL,
            MAX_SHIFT_TIP,
        ),
        // In source-image px, the unit of the projection error's first line.
        (
            cols.offset,
            0,
            egui::DragValue::new(&mut bars.max_projection_error_px)
                .range(0.0..=100.0)
                .speed(0.05)
                .max_decimals(1),
            MAX_PROJECTION_ERROR_LABEL,
            MAX_PROJECTION_ERROR_TIP,
        ),
    ];
    let mut moved = BoxesMoved::default();
    for (x, line_index, value, label, tip) in boxes {
        let min = egui::pos2(
            rect.min.x + x,
            rect.min.y + line_index as f32 * (line + gap),
        );
        let cell = egui::Rect::from_min_max(min, egui::pos2(rect.max.x, min.y + line));
        let mut cell_ui = ui.new_child(
            egui::UiBuilder::new()
                .max_rect(cell)
                .layout(egui::Layout::left_to_right(egui::Align::Center)),
        );
        cell_ui.set_clip_rect(clip);
        moved.merge(bar_box(&mut cell_ui, value, hover));
        cell_ui
            .add_enabled(hover.enabled(), egui::Label::new(label))
            .on_hover_text(tip);
    }
    moved
}
