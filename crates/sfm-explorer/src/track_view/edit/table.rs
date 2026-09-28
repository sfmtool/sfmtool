// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The observation table: the columns, one row per observation, and the
//! three-state verdict control each row carries.
//!
//! The columns Track View has come first -- the rendered
//! patch tile, the image and its name, the reprojection error and the ray angle
//! -- and what the bench adds follows them: the verdict, the stage's own
//! photometric numbers, the kernel's status and where the observation came
//! from. A reader who knows the view-only panel reads this one.
//!
//! Rows are painted at fixed x-offsets rather than laid out by egui, as the
//! view-only panel's are, so the header and every row stay aligned whatever a
//! cell prints, and the header can be drawn above the scroll area rather than
//! as its first row: the offsets are the same either side of that boundary.
//! The one real widget in a row is the verdict control: the row
//! rect is registered first and the control after it, so the control wins the
//! clicks that land on it and the row takes the rest.

use sfmtool_core::bench::{EditableTrack, StageKind, Verdict};
use sfmtool_core::SfmrReconstruction;

use super::{
    measurements, provenance_text, radius_number, row_grids, row_radius, row_surface,
    self_similarity_cell_color, sigma_cell_color, zncc_cell_color, RowGrids, TrackEdit,
    TrackEditResponse,
};
use crate::scene::{ImageRef, ReconId};
use crate::state::AppState;

/// Side of one rendered tile, and so the tallest thing in a row. The size the
/// Track View draws its own tiles at, because they are the same
/// tile.
pub(crate) const TILE_SIZE: f32 = 48.0;
/// Height of one observation row.
pub(crate) const ROW_HEIGHT: f32 = TILE_SIZE + 6.0;
/// Width of the verdict control.
const VERDICT_WIDTH: f32 = 72.0;
/// Side of one cell of a three-by-three grid a row draws: room for the slide
/// line the localizability and the self-similarity grids draw in a cell.
const GRID_CELL: f32 = 10.0;
/// Side of a whole grid: three cells and the four lines of its border.
const GRID_SIDE: f32 = 3.0 * GRID_CELL + 4.0;
/// Side of the self-similarity surface plot a row draws.
const PLOT_SIDE: f32 = 44.0;
/// Side of the same plot in its hover view.
const PLOT_HOVER_SIDE: f32 = 220.0;

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
    /// What the thresholds propose for it, which is what the row is painted by.
    pub painted: Verdict,
    /// The six measurement cells, as printed.
    pub cells: [String; 6],
    /// The three grids drawn beside the ZNCC, the sigma_pos and the
    /// self-similarity cells.
    pub grids: RowGrids,
    /// Whether the row drew a rendered tile, rather than the empty frame that
    /// stands in when there is nothing to render.
    pub tile: bool,
    /// Whether the row drew the self-similarity surface plot.
    pub self_similarity_plot: bool,
}

/// Fixed column x-offsets, relative to the left edge of the table.
pub(super) struct ColumnLayout {
    verdict: f32,
    tile: f32,
    image: f32,
    name: f32,
    zncc: f32,
    zncc_grid: f32,
    shift: f32,
    offset: f32,
    sigma: f32,
    sigma_grid: f32,
    self_similarity: f32,
    self_similarity_grid: f32,
    self_similarity_plot: f32,
    status: f32,
    from: f32,
}

impl ColumnLayout {
    pub(super) fn new() -> Self {
        let verdict = 0.0;
        let tile = verdict + VERDICT_WIDTH + 6.0;
        let image = tile + TILE_SIZE + 8.0;
        let name = image + 34.0;
        let zncc = name + 130.0;
        // Room for the whole-patch and the middle reading in percent,
        // `100 / 100`, then the ZNCC grid.
        let zncc_grid = zncc + 64.0;
        let shift = zncc_grid + GRID_SIDE + 10.0;
        let offset = shift + 62.0;
        // Room for the error in px and in degrees, `12.65 / 0.08`, and for the
        // `Proj. err (px / deg)` heading.
        let sigma = offset + 110.0;
        // Room for the whole and the middle sigma_pos, `0.08 / 0.12`, then
        // the localizability grid.
        let sigma_grid = sigma + 76.0;
        let self_similarity = sigma_grid + GRID_SIDE + 10.0;
        // Room for the whole and the middle radius, `2.3 / 1.4`, then the
        // self-similarity grid.
        let self_similarity_grid = self_similarity + 76.0;
        // Then the core's surface plot.
        let self_similarity_plot = self_similarity_grid + GRID_SIDE + 8.0;
        let status = self_similarity_plot + PLOT_SIDE + 10.0;
        // The status cell holds a sentence at the track stage -- the reason a
        // row was not read, or the walk a fit refused and what it scored -- so
        // it is given room for one and elided to it.
        let from = status + 270.0;
        Self {
            verdict,
            tile,
            image,
            name,
            zncc,
            zncc_grid,
            shift,
            offset,
            sigma,
            sigma_grid,
            self_similarity,
            self_similarity_grid,
            self_similarity_plot,
            status,
            from,
        }
    }

    /// The header's cells, each at the offset its column is drawn at, with the
    /// hover text that says what the column holds.
    pub(super) fn headers(&self) -> [(f32, &'static str, &'static str); 10] {
        [
            (self.verdict, "Verdict", VERDICT_TIP),
            (
                self.image,
                "Img",
                "The image's index in the reconstruction.",
            ),
            (self.name, "Name", "The image's file name."),
            (self.zncc, "ZNCC (%)", ZNCC_TIP),
            (self.shift, "Seed sh.", SEED_SHIFT_TIP),
            (self.offset, "Proj. err (px / deg)", PROJECTION_ERROR_TIP),
            (self.sigma, "\u{3c3}_pos", SIGMA_POS_TIP),
            (self.self_similarity, "Self-sim (px)", SELF_SIMILARITY_TIP),
            (self.status, "Status", STATUS_TIP),
            (self.from, "From", FROM_TIP),
        ]
    }
}

const VERDICT_TIP: &str = "Whether the observation belongs to the track: in, out or \
    candidate. Click a row's control to cycle it; a dot marks a verdict set by hand, which the \
    thresholds leave alone. The tile beside it is the patch as this photograph sees it.";

/// The ZNCC heading's hover text, in one constant so the tests aim at the text
/// shown.
pub(super) const ZNCC_TIP: &str = "Zero-mean normalized cross-correlation, in percent: how \
    closely this photograph's view of the patch matches, once differences in brightness and \
    contrast are removed. 100 is an exact match and 0 is no relation.\n\n\
    The first number is over the whole patch. The second is over its middle only, the centred \
    square half the patch's width, read from the same samples. A high first number with a low \
    second one means the match comes from the patch's surroundings rather than from the \
    pixel's own neighbourhood.\n\n\
    The grid beside them is the same samples read over each ninth of the patch, laid out as the \
    tile is, with every pixel weighted equally: green at 100, yellow at 75, red at 50 and \
    below, grey where the patch is flat. Hover it for the numbers.\n\n\
    At the cluster stage the match is against the reference's template. At the track stage it \
    is against the consensus of the other observations, with this one left out.";

const SEED_SHIFT_TIP: &str = "How far the correlation peak sits from where the observation \
    sits, in source-image pixels: the observation's own evidence. The max shift px bar judges it.";

/// The projection error heading's hover text.
pub(super) const PROJECTION_ERROR_TIP: &str = "The reprojection error, in pixels and then \
    in degrees.\n\n\
    The first number is how far the observation sits from where the track's point projects \
    into this image, in pixels. Before the track is triangulated it is measured to where its \
    patch's centre projects, which is the same place once it is.\n\n\
    The second is the same residual as an angle: between the observation's ray and the \
    direction from its camera to the point, in degrees. It is comparable across lenses and \
    depths, where a pixel is not.\n\n\
    Large errors on every row beside small seed shifts say the point is off, not the \
    sightings. Track stage only.";

/// The sigma_pos heading's hover text.
pub(super) const SIGMA_POS_TIP: &str = "The deprecated localizability score, shown while it is \
    compared with the self-similarity radius beside it.\n\n\
    How precisely this observation's own tile pins a position, as \
    the positional uncertainty in patch-grid pixels. Lower is better; a flat tile or a single \
    straight edge scores high. The max \u{3c3}_pos bar judges the first number.\n\n\
    The second number is the middle of the tile alone, the centred square half its width. The \
    grid beside them is each ninth of the tile alone, laid out as the tile is: green at half \
    the bar and below, red at twice the bar and above. A part has fewer pixels than the whole, \
    so it reads higher for the same texture. A + in a box says that ninth alone pins a position \
    in both directions as well as the bar asks of a whole tile. A line says it pins one \
    direction only, and a match could slide along the line, as along an edge. A box with \
    neither pins nothing. Hover the grid for the numbers.";

/// The self-similarity heading's hover text.
pub(super) const SELF_SIMILARITY_TIP: &str = "The ZNCC self-similarity radius: how far, in \
    patch-grid pixels, this observation's own tile can slide over itself and still match \
    itself as well as a true match between two photographs would: where its ZNCC against \
    itself, interpolated between whole-pixel shifts, falls through that level. \
    Under 1 means a match locks onto this position within a pixel, as on a corner or a busy \
    texture. 3+ means it still matched itself 3 pixels away and may slide further, as along a \
    straight edge or over a flat patch.\n\n\
    The first number is the whole tile, the second its middle alone, the centred square half \
    its width. The grid beside them is each ninth of the tile alone, laid out as the tile is: \
    green under 1, yellow from 1 to 2, orange from 2 to 3, red at 3 or more. A line in a box is \
    the direction that ninth can slide in, where its matching shifts line up along one. Hover \
    the grid for the numbers.";

const STATUS_TIP: &str = "What the last evaluation or fit said about the row. At the \
    cluster stage, the refinement's verdict on the member. At the track stage, localized, a \
    walk the fit refused, or why the row could not be read.";

const FROM_TIP: &str = "Where the observation came from: the point the track was read \
    from, a detected feature, a descriptor search, a sweep, a pixel placed by hand, or another \
    point.";

/// The verdict the control cycles to from `current`: in, out, candidate, round
/// again.
fn next_verdict(current: Verdict) -> Verdict {
    match current {
        Verdict::In => Verdict::Out,
        Verdict::Out => Verdict::Candidate,
        Verdict::Candidate => Verdict::In,
    }
}

/// The word a verdict shows on its control.
fn verdict_text(verdict: Verdict, pinned: bool) -> String {
    let word = match verdict {
        Verdict::In => "in",
        Verdict::Out => "out",
        Verdict::Candidate => "candidate",
    };
    if pinned {
        format!("{word} \u{2022}")
    } else {
        word.to_string()
    }
}

impl TrackEdit {
    /// Draw the observation table and record what it drew.
    pub(super) fn show_table(
        &mut self,
        ui: &mut egui::Ui,
        recon: &SfmrReconstruction,
        id: ReconId,
        state: &AppState,
        track: &EditableTrack,
        response: &mut TrackEditResponse,
    ) {
        let cols = ColumnLayout::new();
        let stage = track.stage_kind();
        let hovered = state
            .hovered_image
            .filter(|i| i.recon == id)
            .map(ImageRef::index);
        let label = self
            .showing
            .as_ref()
            .map(|(_, label)| label.clone())
            .unwrap_or_default();
        let selected = state.selected_bench_observations(id, &label);
        self.rows.clear();

        // Above the scroll area, not inside it: at the bottom of a long track
        // a header that had scrolled away leaves six columns of numbers with
        // nothing saying which is which. Alignment survives the move because
        // every column is left-anchored at the table's left edge, and a scroll
        // area moves its right edge, never that one.
        draw_header(ui, &cols);

        let mut scroll_area = egui::ScrollArea::vertical().auto_shrink([false, false]);
        if let Some(offset) = self.scroll_offset_y {
            scroll_area = scroll_area.vertical_scroll_offset(offset);
        }
        let output = scroll_area.show(ui, |ui| {
            for observation in 0..track.observations.len() {
                self.draw_row(
                    ui,
                    recon,
                    id,
                    state,
                    track,
                    observation,
                    stage,
                    hovered,
                    selected,
                    &cols,
                    response,
                );
            }
        });
        self.scroll_offset_y = Some(output.state.offset.y);
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
        hovered: Option<usize>,
        selected: &[usize],
        cols: &ColumnLayout,
        response: &mut TrackEditResponse,
    ) {
        let row = &track.observations[observation];
        let image = ImageRef::new(id, row.image as usize);
        let painted = self
            .painted
            .get(observation)
            .copied()
            .unwrap_or(row.verdict);
        let cells = measurements(row, stage, &self.evaluation);
        let grids = row_grids(row, stage, &self.evaluation);
        let name = recon
            .image_table
            .images
            .get(image.index())
            .map(|im| im.name.clone())
            .unwrap_or_else(|| format!("#{}", row.image));

        let available = ui.available_rect_before_wrap();
        let rect =
            egui::Rect::from_min_size(available.min, egui::vec2(available.width(), ROW_HEIGHT));
        let row_response = ui.allocate_rect(rect, egui::Sense::click());

        // The painting first, then the selection and the hover over it: what a
        // row would become is a property of the numbers, and which row the
        // pointer or the split is on is a property of this frame.
        let visuals = ui.visuals();
        let paint = match painted {
            Verdict::In => egui::Color32::from_rgb(40, 90, 50),
            Verdict::Out => egui::Color32::from_rgb(96, 44, 44),
            Verdict::Candidate => visuals.faint_bg_color,
        };
        ui.painter()
            .rect_filled(rect, 0.0, paint.gamma_multiply(0.7));
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
        // The row's own menu: what a search runs *from* is one observation, so
        // the gesture is on the row rather than in the toolbar, exactly as the
        // two gestures that name a pixel are in the Image Detail menu rather
        // than here. Registered on the row's rect, so a right-click anywhere in
        // it opens the menu for that observation.
        let label = self
            .showing
            .as_ref()
            .map(|(_, label)| label.clone())
            .unwrap_or_default();
        // When the node's index is absent or out of date the entry itself is
        // the remedy: a person who finds the search greyed has no reason to
        // look anywhere else for it, so the row offers the build in its place.
        // The build does not then run the search -- a build over a large
        // capture takes long enough that the person has moved on, and
        // candidates landing on a track unasked are a surprise.
        let sources = self.build_refusal.as_ref().and_then(|(_, why)| why.clone());
        crate::context_menu::on_secondary_click(&row_response).show(|ui| {
            match state.sift_index_state(id) {
                crate::index_files::IndexFileState::Current => {
                    let button = egui::Button::new(super::SEARCH_DESCRIPTORS_LABEL);
                    let clicked = match state.bench_search_refusal(id, &label, observation) {
                        None => ui
                            .add(button)
                            .on_hover_text(
                                "Ask the SIFT index which other photographs hold the patch \
                                 around this observation, and add each as a candidate",
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
                             and add each match as a candidate",
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
                let clicked = match state.busy_refusal(id) {
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

        if row_response.clicked() {
            response.select_image = Some(image.index());
            // With the image, where in it this observation sits, so the Image
            // Detail panel can bring it into view: the same position its bench
            // layer draws the mark at.
            response.reveal_feature = crate::bench::observation_pixel(row);
            let extend = ui.input(|i| i.modifiers.command || i.modifiers.shift);
            response.pick_row = Some((observation, extend));
        }
        // Camera view for the row's image, as a view-mode row's double-click
        // enters it: the rows of both modes are observations of one track.
        // The observation goes with it, so the view turns to show it.
        if row_response.double_clicked() {
            response.request_camera_view = Some(image.index());
            response.reveal_feature = crate::bench::observation_pixel(row);
        }

        let x0 = rect.min.x;
        let cy = rect.center().y;

        // The verdict control, registered after the row so it takes the clicks
        // that land on it. One button cycling in / out / candidate: three
        // buttons would be three targets for one decision.
        let verdict_rect = egui::Rect::from_min_size(
            egui::pos2(x0 + cols.verdict, cy - 10.0),
            egui::vec2(VERDICT_WIDTH, 20.0),
        );
        let mut verdict_ui = ui.new_child(egui::UiBuilder::new().max_rect(verdict_rect));
        if verdict_ui
            .add(egui::Button::new(verdict_text(row.verdict, row.pinned)).small())
            .on_hover_text("Click to cycle in / out / candidate")
            .clicked()
        {
            response.set_verdict = Some((observation, next_verdict(row.verdict)));
        }

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
        let shown = crate::elide::middle(&name, cols.zncc - cols.name - 8.0, |value| {
            ui.ctx().fonts_mut(|fonts| {
                fonts
                    .layout_no_wrap(value.to_owned(), font.clone(), weak)
                    .rect
                    .width()
            })
        });
        text(cols.name, &shown, weak);
        // While an evaluation of new inputs is on its way the numbers are the
        // last evaluation's, and they are greyed so that they do not read as
        // the numbers of the track as it now stands.
        let number_color = match self.evaluation {
            crate::bench::live::Evaluation::Evaluating => weak,
            _ => text_color,
        };
        for (x, cell) in [
            cols.zncc,
            cols.shift,
            cols.offset,
            cols.sigma,
            cols.self_similarity,
        ]
        .into_iter()
        .zip(cells.iter())
        {
            text(x, cell, number_color);
        }
        // The status cell is a sentence rather than a number at the track
        // stage, so it is elided to its column the way the image name is.
        let status = crate::elide::middle(&cells[5], cols.from - cols.status - 8.0, |value| {
            ui.ctx().fonts_mut(|fonts| {
                fonts
                    .layout_no_wrap(value.to_owned(), font.clone(), text_color)
                    .rect
                    .width()
            })
        });
        text(cols.status, &status, text_color);
        text(cols.from, &provenance_text(row.provenance), weak);

        // The three grids, faded with the numbers while an evaluation is on its
        // way. Each is laid out as the tile is, so a cell sits over the part
        // of the tile it read.
        let fade = if number_color == weak { 0.45 } else { 1.0 };
        let bar = track.thresholds.max_keypoint_uncertainty;
        let drawn = [
            (
                cols.zncc_grid,
                grids.zncc,
                None,
                &zncc_cell_color as &dyn Fn(f64) -> Option<egui::Color32>,
                GridKind::Zncc,
            ),
            (
                cols.sigma_grid,
                grids.sigma,
                grids.sigma.zip(grids.slide).map(|(sigma, slide)| {
                    std::array::from_fn(|row| {
                        std::array::from_fn(|col| cell_mark(sigma[row][col], slide[row][col], bar))
                    })
                }),
                &|sigma| sigma_cell_color(sigma, bar),
                GridKind::Sigma,
            ),
            (
                cols.self_similarity_grid,
                grids.radius,
                grids
                    .radius_slide
                    .map(|slide| slide.map(|row| row.map(slide_mark))),
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

        // The core's self-similarity surface, with the contour its radius is
        // read at, and a larger one on hover.
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
            painted,
            cells,
            grids,
            tile: tile.is_some(),
            self_similarity_plot: plotted,
        });
    }
}

/// The words under the hover view of a surface plot: the radius and the
/// level the contour is drawn at.
pub(super) fn plot_caption(radius: Option<f64>, tolerance: f64) -> String {
    let radius = radius.map_or_else(|| "-".to_string(), super::radius_number);
    format!(
        "ZNCC of the patch against itself at every shift up to 3 px. The contour is at \
         {:.3}, one minus the tolerance {:.3}; the dots are the whole-pixel shifts inside \
         it. Radius {radius}.",
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
                CellMark::Plus => {
                    let arm = GRID_CELL / 2.0 - 1.5;
                    painter
                        .line_segment([c - egui::vec2(arm, 0.0), c + egui::vec2(arm, 0.0)], stroke);
                    painter
                        .line_segment([c - egui::vec2(0.0, arm), c + egui::vec2(0.0, arm)], stroke);
                }
                CellMark::Line(half) => {
                    painter.line_segment([c - half, c + half], stroke);
                }
            }
        }
    }
}

/// What a localizability grid cell draws over its colour: whether that ninth
/// of the tile pins a position in both directions, in one, or in neither.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) enum CellMark {
    /// Neither direction is pinned; the colour says so already.
    Nothing,
    /// Both directions are pinned.
    Plus,
    /// Only the direction across the line is pinned, and a match could slide
    /// along it: the half line from the cell's centre, in screen points. The
    /// grid frame's `x` is the screen's right and its `y` the screen's down,
    /// so the slide is drawn as it is.
    Line(egui::Vec2),
}

/// The [`CellMark`] of a cell with localizability `sigma` and slide vector
/// `slide`, against `bar`, the track's max sigma_pos.
///
/// `sigma` is the uncertainty along the cell's weak axis. The strong axis's is
/// `sigma * sqrt(1 - |slide|)`, since `|slide|` is `1 - λ₂/λ₁` and each
/// uncertainty goes as one over the root of its eigenvalue. An axis counts as
/// pinned when its uncertainty is at or under the bar, the bar a whole tile is
/// judged by: a `+` when the weak axis is pinned, and so both are; a line along
/// the slide when only the strong axis is; nothing when neither is.
pub(super) fn cell_mark(sigma: f64, slide: [f64; 2], bar: f64) -> CellMark {
    let strength = slide[0].hypot(slide[1]);
    if !sigma.is_finite() || !strength.is_finite() {
        return CellMark::Nothing;
    }
    // A bar at 0 turns every row out; the marks still need one to judge by.
    let bar = if bar > 0.0 {
        bar
    } else {
        sfmtool_core::bench::Thresholds::default().max_keypoint_uncertainty
    };
    if sigma <= bar {
        return CellMark::Plus;
    }
    let strong = sigma * (1.0 - strength.min(1.0)).sqrt();
    if strong > bar || strength == 0.0 {
        return CellMark::Nothing;
    }
    let along = egui::vec2(slide[0] as f32, slide[1] as f32) / strength as f32;
    CellMark::Line(along * (GRID_CELL / 2.0 - 1.0))
}

/// The [`CellMark`] of a self-similarity cell with slide vector `slide`: a
/// line along the slide where it is at least `0.5` long, so the cell's
/// matching shifts line up along one direction, and nothing otherwise.
pub(super) fn slide_mark(slide: [f64; 2]) -> CellMark {
    let strength = slide[0].hypot(slide[1]);
    if !strength.is_finite() || strength < 0.5 {
        return CellMark::Nothing;
    }
    let along = egui::vec2(slide[0] as f32, slide[1] as f32) / strength as f32;
    CellMark::Line(along * (GRID_CELL / 2.0 - 1.0))
}

/// Which of a row's grids a value belongs to, which says how its hover text
/// prints it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) enum GridKind {
    /// A ZNCC, printed in percent.
    Zncc,
    /// A sigma_pos, printed to two decimals.
    Sigma,
    /// A self-similarity radius, printed as the cell prints it.
    SelfSimilarity,
}

/// A grid's nine values as its hover text shows them, three to a line: a
/// ZNCC in percent, a sigma_pos to two decimals, a self-similarity radius as
/// its cell prints it (`1.4`, `3+`), and `-` for a cell with no reading.
pub(super) fn grid_numbers(grid: &[[f64; 3]; 3], kind: GridKind) -> String {
    grid.iter()
        .map(|row| {
            row.iter()
                .map(|&value| match (value.is_finite(), kind) {
                    (false, _) => format!("{:>5}", "-"),
                    (true, GridKind::Zncc) => format!("{:>5.0}", 100.0 * value),
                    (true, GridKind::Sigma) => format!("{value:>5.2}"),
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
        "Move this sighting {:.1} px to ({:.1}, {:.1}), where the last fit's walk \
         would have put it. ZNCC {} at the seed, {} at the walked peak. Pins it, \
         as a hand placement does.",
        m.walked_px.unwrap_or(f64::NAN),
        to[0],
        to[1],
        zncc(m.zncc, m.zncc_middle),
        zncc(m.walked_zncc, m.walked_zncc_middle),
    ))
}

/// The header row, at the same offsets the rows draw at, drawn once above the
/// scroll area so it stays put while the rows move under it.
fn draw_header(ui: &mut egui::Ui, cols: &ColumnLayout) {
    let available = ui.available_rect_before_wrap();
    let rect = egui::Rect::from_min_size(available.min, egui::vec2(available.width(), 20.0));
    ui.allocate_rect(rect, egui::Sense::hover());
    let font = egui::TextStyle::Small.resolve(ui.style());
    let color = ui.visuals().weak_text_color();
    let headers = cols.headers();
    for (k, &(x, label, tip)) in headers.iter().enumerate() {
        ui.painter().text(
            egui::pos2(rect.min.x + x, rect.center().y),
            egui::Align2::LEFT_CENTER,
            label,
            font.clone(),
            color,
        );
        // The heading's hover region runs to where the next one starts, so the
        // whole width of its column answers, not only the word.
        let end = headers
            .get(k + 1)
            .map_or(rect.max.x, |&(next, _, _)| rect.min.x + next);
        let cell = egui::Rect::from_x_y_ranges(rect.min.x + x..=end, rect.y_range());
        ui.interact(
            cell,
            ui.id().with(("track_view_heading", k)),
            egui::Sense::hover(),
        )
        .on_hover_text(tip);
    }
}
