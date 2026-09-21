// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The observation table: the columns, one row per observation, and the
//! three-state verdict control each row carries.
//!
//! The columns the Point Track Detail panel has come first -- the rendered
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

use super::{measurements, provenance_text, TrackEdit, TrackEditResponse};
use crate::scene::{ImageRef, ReconId};
use crate::state::AppState;

/// Side of one rendered tile, and so the tallest thing in a row. The size the
/// Point Track Detail panel draws its own tiles at, because they are the same
/// tile.
pub(crate) const TILE_SIZE: f32 = 48.0;
/// Height of one observation row.
pub(crate) const ROW_HEIGHT: f32 = TILE_SIZE + 6.0;
/// Width of the verdict control.
const VERDICT_WIDTH: f32 = 72.0;

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
    /// The seven measurement cells, as printed.
    pub cells: [String; 7],
    /// Whether the row drew a rendered tile, rather than the empty frame that
    /// stands in when there is nothing to render.
    pub tile: bool,
}

/// Fixed column x-offsets, relative to the left edge of the table.
pub(super) struct ColumnLayout {
    verdict: f32,
    tile: f32,
    image: f32,
    name: f32,
    zncc: f32,
    shift: f32,
    offset: f32,
    sigma: f32,
    error: f32,
    angle: f32,
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
        let shift = zncc + 54.0;
        let offset = shift + 62.0;
        let sigma = offset + 62.0;
        let error = sigma + 56.0;
        let angle = error + 54.0;
        let status = angle + 54.0;
        // The status cell holds a sentence at the track stage -- the reason a
        // row was not read -- so it is given room for one and elided to it.
        let from = status + 190.0;
        Self {
            verdict,
            tile,
            image,
            name,
            zncc,
            shift,
            offset,
            sigma,
            error,
            angle,
            status,
            from,
        }
    }

    /// The header's cells, each at the offset its column is drawn at.
    pub(super) fn headers(&self) -> [(f32, &'static str); 11] {
        [
            (self.verdict, "Verdict"),
            (self.image, "Img"),
            (self.name, "Name"),
            (self.zncc, "ZNCC"),
            (self.shift, "Seed sh."),
            (self.offset, "Proj. off"),
            (self.sigma, "\u{3c3}_pos"),
            (self.error, "Error"),
            (self.angle, "Angle"),
            (self.status, "Status"),
            (self.from, "From"),
        ]
    }
}

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
        let cells = measurements(row, stage);
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
        if self.selected_rows.contains(&observation) {
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
            .selection_of
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
                crate::sift_index::SiftIndexState::Current => {
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
                kind => {
                    let stale = kind == crate::sift_index::SiftIndexState::Stale;
                    let text = match stale {
                        true => super::REBUILD_INDEX_TO_SEARCH,
                        false => super::BUILD_INDEX_TO_SEARCH,
                    };
                    // The staleness sentence where there is one, so the entry
                    // says what is wrong as well as what to do about it.
                    let hint = state
                        .sift_index(id)
                        .and_then(|index| index.stale_reason())
                        .map(str::to_string)
                        .unwrap_or_else(|| {
                            "Index every .sift file of this reconstruction into a .kdf beside \
                             its .sfmr. Runs on a worker thread, and does not then run the \
                             search."
                                .to_string()
                        });
                    let refusal = state
                        .busy_refusal(id)
                        .or_else(|| state.sift_index_home_refusal(id))
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
                        response.build_sift_index = true;
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
                            "Project this track's surfel into the reconstruction's other cameras, \
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
        });

        if row_response.clicked() {
            response.select_image = Some(image.index());
            // With the image, where in it this observation sits, so the Image
            // Detail panel can bring it into view: the same position its bench
            // layer draws the mark at.
            response.reveal_feature = crate::bench::observation_pixel(row);
            let extend = ui.input(|i| i.modifiers.command || i.modifiers.shift);
            self.toggle_row(observation, extend);
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
        for (x, cell) in [
            cols.zncc,
            cols.shift,
            cols.offset,
            cols.sigma,
            cols.error,
            cols.angle,
        ]
        .into_iter()
        .zip(cells.iter())
        {
            text(x, cell, text_color);
        }
        // The status cell is a sentence rather than a number at the track
        // stage, so it is elided to its column the way the image name is.
        let status = crate::elide::middle(&cells[6], cols.from - cols.status - 8.0, |value| {
            ui.ctx().fonts_mut(|fonts| {
                fonts
                    .layout_no_wrap(value.to_owned(), font.clone(), text_color)
                    .rect
                    .width()
            })
        });
        text(cols.status, &status, text_color);
        text(cols.from, &provenance_text(row.provenance), weak);

        self.rows.push(RowSummary {
            observation,
            image: row.image,
            verdict: row.verdict,
            pinned: row.pinned,
            painted,
            cells,
            tile: tile.is_some(),
        });
    }

    /// Add or remove one row from the selection a split reads.
    ///
    /// A plain click selects that row alone; Ctrl or Shift extends, which is
    /// what picking out the observations of the other surface takes.
    fn toggle_row(&mut self, observation: usize, extend: bool) {
        if !extend {
            self.selected_rows = vec![observation];
            return;
        }
        match self.selected_rows.iter().position(|&r| r == observation) {
            Some(at) => {
                self.selected_rows.remove(at);
            }
            None => {
                self.selected_rows.push(observation);
                self.selected_rows.sort_unstable();
            }
        }
    }
}

/// The header row, at the same offsets the rows draw at, drawn once above the
/// scroll area so it stays put while the rows move under it.
fn draw_header(ui: &mut egui::Ui, cols: &ColumnLayout) {
    let available = ui.available_rect_before_wrap();
    let rect = egui::Rect::from_min_size(available.min, egui::vec2(available.width(), 20.0));
    ui.allocate_rect(rect, egui::Sense::hover());
    let painter = ui.painter();
    let font = egui::TextStyle::Small.resolve(ui.style());
    let color = ui.visuals().weak_text_color();
    for (x, label) in cols.headers() {
        painter.text(
            egui::pos2(rect.min.x + x, rect.center().y),
            egui::Align2::LEFT_CENTER,
            label,
            font.clone(),
            color,
        );
    }
}
