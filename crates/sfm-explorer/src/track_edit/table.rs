// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The observation table: the columns, one row per observation, and the
//! three-state verdict control each row carries.
//!
//! The columns the Point Track Detail panel has come first -- the thumbnail,
//! the image and its name, the reprojection error and the ray angle -- and what
//! the bench adds follows them: the verdict, the stage's own photometric
//! numbers, the kernel's status and where the observation came from. A reader
//! who knows the view-only panel reads this one.
//!
//! Rows are painted at fixed x-offsets rather than laid out by egui, as the
//! view-only panel's are, so the header and every row stay aligned whatever a
//! cell prints. The one real widget in a row is the verdict control: the row
//! rect is registered first and the control after it, so the control wins the
//! clicks that land on it and the row takes the rest.

use sfmtool_core::bench::{EditableTrack, StageKind, Verdict};
use sfmtool_core::SfmrReconstruction;

use super::{measurements, provenance_text, TrackEdit, TrackEditResponse};
use crate::scene::{ImageRef, ReconId};
use crate::state::AppState;

/// Height of one thumbnail, and so of the tallest thing in a row.
pub(crate) const THUMB_SIZE: f32 = 40.0;
/// Height of one observation row.
pub(crate) const ROW_HEIGHT: f32 = THUMB_SIZE + 6.0;
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
    /// The six measurement cells, as printed.
    pub cells: [String; 6],
}

/// Fixed column x-offsets, relative to the left edge of the table.
struct ColumnLayout {
    verdict: f32,
    thumb: f32,
    image: f32,
    name: f32,
    zncc: f32,
    shift: f32,
    sigma: f32,
    error: f32,
    angle: f32,
    status: f32,
    from: f32,
}

impl ColumnLayout {
    fn new() -> Self {
        let verdict = 0.0;
        let thumb = verdict + VERDICT_WIDTH + 6.0;
        let image = thumb + THUMB_SIZE + 8.0;
        let name = image + 34.0;
        let zncc = name + 150.0;
        let shift = zncc + 54.0;
        let sigma = shift + 54.0;
        let error = sigma + 56.0;
        let angle = error + 54.0;
        let status = angle + 54.0;
        let from = status + 104.0;
        Self {
            verdict,
            thumb,
            image,
            name,
            zncc,
            shift,
            sigma,
            error,
            angle,
            status,
            from,
        }
    }

    /// The header's cells, each at the offset its column is drawn at.
    fn headers(&self) -> [(f32, &'static str); 10] {
        [
            (self.verdict, "Verdict"),
            (self.image, "Img"),
            (self.name, "Name"),
            (self.zncc, "ZNCC"),
            (self.shift, "Shift"),
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

        let mut scroll_area = egui::ScrollArea::vertical().auto_shrink([false, false]);
        if let Some(offset) = self.scroll_offset_y {
            scroll_area = scroll_area.vertical_scroll_offset(offset);
        }
        let output = scroll_area.show(ui, |ui| {
            draw_header(ui, &cols);
            for observation in 0..track.observations.len() {
                self.draw_row(
                    ui,
                    recon,
                    id,
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

    /// One observation: its painting, its verdict control, its thumbnail and
    /// its cells.
    #[allow(clippy::too_many_arguments)]
    fn draw_row(
        &mut self,
        ui: &mut egui::Ui,
        recon: &SfmrReconstruction,
        id: ReconId,
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
        if row_response.clicked() {
            response.select_image = Some(image.index());
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

        if image.index() < recon.image_table.thumbnails_y_x_rgb.shape()[0] {
            if !self.thumbnail_textures.contains_key(&image) {
                self.load_thumbnail(ui.ctx(), recon, image);
            }
            if let Some(texture) = self.thumbnail_textures.get(&image) {
                let thumb = egui::Rect::from_min_size(
                    egui::pos2(x0 + cols.thumb, cy - THUMB_SIZE / 2.0),
                    egui::vec2(THUMB_SIZE, THUMB_SIZE),
                );
                ui.painter().image(
                    texture.id(),
                    thumb,
                    egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                    egui::Color32::WHITE,
                );
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
            cols.sigma,
            cols.error,
            cols.angle,
            cols.status,
        ]
        .into_iter()
        .zip(cells.iter())
        {
            text(x, cell, text_color);
        }
        text(cols.from, &provenance_text(row.provenance), weak);

        self.rows.push(RowSummary {
            observation,
            image: row.image,
            verdict: row.verdict,
            pinned: row.pinned,
            painted,
            cells,
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

/// The header row, at the same offsets the rows draw at.
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
