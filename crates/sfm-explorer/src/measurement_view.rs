// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The Measurement View panel: one measurement of the selected reconstruction,
//! in the way Track View shows one track.
//!
//! See `specs/gui/measurement-view.md`. The panel opens with Track View's
//! **Edit** box, ticked exactly when the focused item is a measurement on the
//! selected node's bench. With it clear, the panel shows the selected stored
//! measurement, read-only: its header, then the body for its kind. With it
//! ticked, it shows the focused item instead, with a toolbar (*Commit*,
//! *Discard*, *Rename*), fields that set its label and stated distance, and
//! the item's **live readings**, read from the item rather than from a stored
//! row. A distance measurement's body is its two
//! endpoint blocks side by side, `A` and `B`, then the result block. Each
//! block shows the endpoint's photograph and pixel, a crop of the photograph
//! around it with a larger view on hover, the stored patch of the point it
//! resolves to with a cross at its texel, and how it was resolved.
//!
//! An endpoint block takes Track View's row gestures: a click selects the
//! endpoint and its image and reveals its pixel in Image Detail, a
//! double-click enters camera view through its image turned toward the pixel,
//! and a hover sets the cross-panel hover to its image. The block's own
//! buttons keep their clicks.
//!
//! The panel decides nothing. What a person does lands in
//! [`MeasurementViewResponse`], and the dock applies it through `AppState`.
//! A stored measurement's reading is the node's cached one
//! ([`crate::measurements::readings`]); an item's is kept here per item value
//! and document serial.

use std::collections::HashMap;
use std::sync::Arc;

use egui::{Color32, Pos2, Rect, Stroke};
use sfmtool_core::bench::EditableMeasurement;
use sfmtool_core::camera::TiledPhotograph;
use sfmtool_core::reconstruction::measurement_endpoint::{
    CandidateStanding, EndpointPosition, EndpointRefusal, EndpointResolution,
    ResolvedDistanceMeasurement,
};
use sfmtool_core::LengthUnit;

use crate::document::VersionSerial;
use crate::measurements::{
    self, DistanceResult, DistanceRow, EndpointSide, ItemReading, MeasurementRef, MINUS,
};
use crate::patch_pictures::crop::{self, DrawnCrop, Outline, CELL_TEXTURE_OPTIONS};
use crate::patch_pictures::marks::CONTEXT_HOVER_SIDE;
use crate::patch_pictures::PanelRegionPool;
use crate::scene::{ImageRef, ReconId, SceneNode};
use crate::state::{AppState, DisplayPhotograph};

#[cfg(test)]
mod tests;

/// The side of the square of photograph a crop shows around an endpoint no
/// square holds, in photograph pixels.
pub(crate) const UNRESOLVED_CROP_PX: f64 = 64.0;

/// The side of an endpoint's crop in the panel, in points.
const CROP_SIDE: f32 = 128.0;

/// The side of an endpoint's patch picture in the panel, in points.
const PATCH_SIDE: f32 = 64.0;

/// The empty state's first line with no measurement selected.
pub(crate) const NO_MEASUREMENT: &str = "No measurement selected";

/// The *Edit* box's label, Track View's word.
pub(crate) const EDIT_LABEL: &str = "Edit";

/// The name of the field *Rename* opens.
pub(crate) const RENAME_NAME: &str = "measurement item name";

/// What one frame of the panel asks the dock to do.
#[derive(Debug, Default, Clone, PartialEq)]
pub(crate) struct MeasurementViewResponse {
    /// Whether the pointer is over the panel, which makes it the owner of
    /// the hover fields for the frame.
    pub(crate) has_pointer: bool,
    /// The image of the endpoint block under the pointer.
    pub(crate) hovered_image: Option<ImageRef>,
    /// A measurement link of the empty state was clicked.
    pub(crate) select_measurement: Option<MeasurementRef>,
    /// An endpoint block of a stored measurement was clicked: select the
    /// endpoint and its image, and reveal its pixel in Image Detail.
    pub(crate) select_endpoint: Option<(MeasurementRef, EndpointSide)>,
    /// An endpoint block of a stored measurement was double-clicked: enter
    /// camera view through its image, turned toward its pixel.
    pub(crate) look_through: Option<(MeasurementRef, EndpointSide)>,
    /// An endpoint block of the focused item on this node was clicked.
    pub(crate) select_item_endpoint: Option<(ReconId, EndpointSide)>,
    /// An endpoint block of the focused item on this node was double-clicked.
    pub(crate) look_through_item: Option<(ReconId, EndpointSide)>,
    /// *Go to Point* beside an endpoint's point was clicked: open the Go to
    /// Point dialog with this ID in it.
    pub(crate) goto_point: Option<String>,
    /// The *Edit* box was ticked (`true`) or cleared on this node.
    pub(crate) set_edit: Option<(ReconId, bool)>,
    /// *Commit* was clicked on the item with this label.
    pub(crate) commit: Option<(ReconId, String)>,
    /// *Discard* was clicked on the item with this label.
    pub(crate) discard: Option<(ReconId, String)>,
    /// *Rename* was committed: the item, and the label it should take.
    pub(crate) rename: Option<(ReconId, String, String)>,
    /// The label field was committed: the item, and the label it should
    /// hold.
    pub(crate) set_label: Option<(ReconId, String, String)>,
    /// The stated distance field was committed: the item, and the text typed.
    pub(crate) set_distance: Option<(ReconId, String, String)>,
}

/// One of the edited mode header's fields.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum HeaderField {
    /// The measurement's own label.
    Label,
    /// The stated distance, a number and a unit.
    Distance,
}

impl HeaderField {
    fn index(self) -> usize {
        match self {
            HeaderField::Label => 0,
            HeaderField::Distance => 1,
        }
    }

    /// The field's name, in the widget listing.
    pub(crate) fn widget_name(self) -> &'static str {
        match self {
            HeaderField::Label => LABEL_FIELD_NAME,
            HeaderField::Distance => DISTANCE_FIELD_NAME,
        }
    }
}

/// The name of the edited mode's label field.
pub(crate) const LABEL_FIELD_NAME: &str = "measurement label";

/// The name of the edited mode's stated distance field.
pub(crate) const DISTANCE_FIELD_NAME: &str = "measurement stated distance";

/// A header field being typed in: which item and field, the text the field
/// held when it took focus, and the text typed so far.
struct FieldDraft {
    item: String,
    field: HeaderField,
    original: String,
    text: String,
}

/// Which picture of which endpoint of which version a cached picture is: the
/// endpoint's image and pixel, which is all a picture is cut from, so a
/// stored row and an item with the same endpoint share one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct PictureKey {
    node: ReconId,
    document: VersionSerial,
    side: EndpointSide,
    image: u32,
    pixel: [u64; 2],
}

impl PictureKey {
    fn new(
        node: ReconId,
        document: VersionSerial,
        side: EndpointSide,
        image: u32,
        pixel: [f64; 2],
    ) -> Self {
        Self {
            node,
            document,
            side,
            image,
            pixel: pixel.map(f64::to_bits),
        }
    }
}

/// An endpoint's crop, cut once its photograph is decoded.
enum Crop {
    /// The crop in its cell and in its hover view, with the caption the hover
    /// view gives.
    Ready {
        cell: Box<DrawnCrop>,
        context: Option<Box<DrawnCrop>>,
        caption: String,
    },
    /// Why there is no crop.
    Absent(&'static str),
}

/// What the panel keeps of one endpoint's pictures between frames.
struct EndpointPictures {
    crop: Crop,
    /// The chosen point's stored patch, `None` where the endpoint does not
    /// resolve or the point stores no bitmap.
    patch: Option<egui::TextureHandle>,
}

/// The live readings of the focused item, kept against the item value and
/// the document serial they were read at, so a step on the item, or an edit
/// of the reconstruction under it, reads them again on the next frame.
struct HeldReading {
    item: Arc<EditableMeasurement>,
    document: VersionSerial,
    reading: ItemReading,
}

/// Measurement View panel state: the pictures of the shown measurement's
/// endpoints, the focused item's readings, a rename in progress, and the
/// region pool its crops are read through.
#[derive(Default)]
pub(crate) struct MeasurementView {
    regions: PanelRegionPool,
    pictures: HashMap<PictureKey, EndpointPictures>,
    reading: Option<HeldReading>,
    /// A rename in progress: the item, and the text typed so far.
    renaming: Option<(String, String)>,
    /// A header field being typed in.
    draft: Option<FieldDraft>,
    /// Where each header field was drawn on the last frame, for the tests
    /// that click one.
    fields: [Option<Rect>; 2],
    /// Where each endpoint block's name was drawn on the last frame, for the
    /// tests that click one.
    names: [Option<Rect>; 2],
    /// Where each measurement link of the empty state was drawn on the last
    /// frame, for the tests that click one.
    links: Vec<Rect>,
    /// What the toolbar drew on the last frame, for the tests that read it.
    toolbar: Vec<(String, Option<String>)>,
}

impl MeasurementView {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Where the name line of the endpoint block on `side` was drawn on the
    /// last frame.
    #[cfg(test)]
    pub(crate) fn name_rect(&self, side: EndpointSide) -> Option<Rect> {
        self.names[side.index()]
    }

    /// Where the edited mode header's `field` was drawn on the last frame.
    #[cfg(test)]
    pub(crate) fn field_rect(&self, field: HeaderField) -> Option<Rect> {
        self.fields[field.index()]
    }

    /// Where the empty state's measurement links were drawn on the last frame.
    #[cfg(test)]
    pub(crate) fn link_rects(&self) -> &[Rect] {
        &self.links
    }

    /// The toolbar's buttons as the last frame drew them, each with the
    /// refusal it was greyed with, if any. Empty with the *Edit* box clear.
    #[cfg(test)]
    pub(crate) fn toolbar(&self) -> &[(String, Option<String>)] {
        &self.toolbar
    }

    /// The focused item's readings, as the last frame read them.
    #[cfg(test)]
    pub(crate) fn item_reading(&self) -> Option<&ItemReading> {
        self.reading.as_ref().map(|held| &held.reading)
    }

    /// Draw the panel and report what the person did with it.
    pub(crate) fn show(&mut self, ui: &mut egui::Ui, state: &AppState) -> MeasurementViewResponse {
        let mut response = MeasurementViewResponse::default();
        if let Some(pos) = ui.input(|i| i.pointer.hover_pos()) {
            response.has_pointer = ui.available_rect_before_wrap().contains(pos);
        }
        self.names = [None, None];
        self.fields = [None, None];
        self.links.clear();
        self.toolbar.clear();
        let Some(node) = crate::scene::selected_node(&state.scene, state.selected_recon) else {
            self.pictures.clear();
            self.reading = None;
            ui.centered_and_justified(|ui| {
                ui.label("No reconstruction loaded");
            });
            return response;
        };
        let rows = measurements::distance_rows(node.edited().measurements());
        let readings = measurements::readings(node);
        let selected = state
            .selected_measurement
            .filter(|m| m.recon == node.id && m.index() < rows.len());
        let editing = state.focused_measurement_label(node.id).map(str::to_string);
        show_edit_box(
            ui,
            state,
            node.id,
            editing.is_some(),
            selected,
            &mut response,
        );
        ui.separator();
        if let Some(label) = editing {
            self.show_item(
                ui,
                state,
                node,
                &label,
                &rows,
                &readings.distance,
                &mut response,
            );
            return response;
        }
        self.reading = None;
        self.renaming = None;
        self.draft = None;
        let Some(measurement) = selected else {
            self.pictures.clear();
            self.show_empty_state(ui, node, &rows, &readings.distance, &mut response);
            return response;
        };
        let row = &rows[measurement.index()];
        let Some(resolved) = readings.distance.get(measurement.index()) else {
            return response;
        };
        let unit = node.recon().metadata.world_space_unit.as_deref();
        let blocks = EndpointSide::BOTH.map(|side| {
            Some(Endpoint {
                image: row.image(side),
                pixel: row.pixel(side),
                resolution: measurements::endpoint(resolved, side).clone(),
            })
        });
        let target = Target::Stored(measurement);
        egui::ScrollArea::vertical()
            .id_salt("measurement_view")
            .auto_shrink([false, false])
            .show(ui, |ui| {
                show_header(ui, node, measurement, row, &rows, &readings.distance);
                ui.separator();
                self.show_blocks(ui, state, node, target, &blocks, &mut response);
                ui.separator();
                show_result(ui, row, resolved, unit);
            });
        response
    }

    /// Edited mode: the focused item `label` on `node`'s bench, with the
    /// toolbar, its header and its endpoints read live.
    #[allow(
        clippy::too_many_arguments,
        reason = "the frame's inputs, read once by show"
    )]
    fn show_item(
        &mut self,
        ui: &mut egui::Ui,
        state: &AppState,
        node: &SceneNode,
        label: &str,
        rows: &[DistanceRow],
        resolved: &[ResolvedDistanceMeasurement],
        response: &mut MeasurementViewResponse,
    ) {
        let Some(item) = state.bench_measurement(node.id, label).cloned() else {
            return;
        };
        let document = node.history.current_version().document_serial;
        let stale = self
            .reading
            .as_ref()
            .is_none_or(|held| !Arc::ptr_eq(&held.item, &item) || held.document != document);
        if stale {
            self.reading = Some(HeldReading {
                item: Arc::clone(&item),
                document,
                reading: measurements::item_reading(node.edited(), &item),
            });
        }
        let reading = self
            .reading
            .as_ref()
            .map(|held| held.reading.clone())
            .expect("just read");
        let origin = state.resolved_measurement_origin(node, &item);
        let unit = node.recon().metadata.world_space_unit.as_deref();
        let endpoints = item.as_distance().map_or([None, None], |d| d.endpoints);
        let blocks: [Option<Endpoint>; 2] = [0, 1].map(|k| {
            let endpoint = endpoints[k]?;
            Some(Endpoint {
                image: endpoint.image,
                pixel: endpoint.pixel,
                resolution: reading.endpoints[k].clone()?,
            })
        });
        let row = measurements::item_row(&item);
        egui::ScrollArea::vertical()
            .id_salt("measurement_view_item")
            .auto_shrink([false, false])
            .show(ui, |ui| {
                self.show_toolbar(ui, state, node.id, label, response);
                ui.separator();
                show_item_origin(ui, node, label, &item, origin);
                self.show_header_fields(ui, state, node, label, &item, response);
                show_item_median(ui, row.as_ref(), || {
                    let row = row.as_ref()?;
                    let reading = reading.resolved.as_ref()?;
                    measurements::item_scale_comparison(rows, resolved, origin, row, reading)
                });
                ui.separator();
                self.show_blocks(ui, state, node, Target::Item(node.id), &blocks, response);
                ui.separator();
                match (&row, &reading.resolved) {
                    (Some(row), Some(resolved)) => show_result(ui, row, resolved, unit),
                    _ => {
                        ui.label(egui::RichText::new("Result").strong());
                        ui.label(item_without_result(&item));
                    }
                }
            });
    }

    /// The edited mode header's two fields: the measurement's own label and
    /// its stated distance, each greyed with the refusal a step would give
    /// while a task holds the node or the bench is view-only.
    fn show_header_fields(
        &mut self,
        ui: &mut egui::Ui,
        state: &AppState,
        node: &SceneNode,
        label: &str,
        item: &EditableMeasurement,
        response: &mut MeasurementViewResponse,
    ) {
        let refusal = state.bench_edit_refusal(node.id);
        let bare =
            crate::bench::measurement::bare_unit(node.recon().metadata.world_space_unit.as_deref());
        let stated = item
            .as_distance()
            .and_then(|d| d.distance)
            .map_or(String::new(), |(distance, unit)| {
                format!("{} {}", measurements::length_number(distance), unit.name())
            });
        egui::Grid::new("measurement_view_header_fields")
            .num_columns(2)
            .show(ui, |ui| {
                ui.label("Label");
                if let Some(to) = self.header_field(
                    ui,
                    label,
                    HeaderField::Label,
                    &item.label,
                    refusal.as_deref(),
                    "No label",
                    "The measurement's own label, written into the file on commit. Enter or \
                     clicking elsewhere sets it; Escape puts it back.",
                ) {
                    response.set_label = Some((node.id, label.to_string(), to));
                }
                ui.end_row();
                ui.label("Stated distance");
                let hint = format!(
                    "A number and a unit: 12in, 30.48 cm. A bare number is in {}. Enter or \
                     clicking elsewhere sets it; Escape puts it back.",
                    bare.name()
                );
                if let Some(to) = self.header_field(
                    ui,
                    label,
                    HeaderField::Distance,
                    &stated,
                    refusal.as_deref(),
                    "No stated distance",
                    &hint,
                ) {
                    response.set_distance = Some((node.id, label.to_string(), to));
                }
                ui.end_row();
            });
    }

    /// One header field of the item `item`, holding `value` while nobody is
    /// typing in it. What it was set to, on the frame it is committed: Enter,
    /// or the field losing focus, with text other than the text it held when
    /// it took focus. Escape puts the field back and commits nothing.
    #[allow(
        clippy::too_many_arguments,
        reason = "the field's own inputs, each one read once"
    )]
    fn header_field(
        &mut self,
        ui: &mut egui::Ui,
        item: &str,
        field: HeaderField,
        value: &str,
        refusal: Option<&str>,
        empty: &str,
        hint: &str,
    ) -> Option<String> {
        let mine = |draft: &FieldDraft| draft.item == item && draft.field == field;
        let mut text = match &self.draft {
            Some(draft) if mine(draft) => draft.text.clone(),
            _ => value.to_string(),
        };
        let edit = egui::TextEdit::singleline(&mut text)
            .desired_width(180.0)
            .hint_text(empty);
        let shown = ui.add_enabled(refusal.is_none(), edit);
        crate::widget_name::set_widget_name(&shown, field.widget_name());
        self.fields[field.index()] = Some(shown.rect);
        let shown = match refusal {
            Some(why) => shown.on_disabled_hover_text(why),
            None => shown.on_hover_text(hint),
        };
        if shown.lost_focus() {
            let cancelled = ui.input(|i| i.key_pressed(egui::Key::Escape));
            let original = match self.draft.take() {
                Some(draft) if mine(&draft) => draft.original,
                other => {
                    self.draft = other;
                    value.to_string()
                }
            };
            return (!cancelled && text != original).then_some(text);
        }
        if shown.has_focus() {
            match &mut self.draft {
                Some(draft) if mine(draft) => draft.text = text,
                _ => {
                    self.draft = Some(FieldDraft {
                        item: item.to_string(),
                        field,
                        original: value.to_string(),
                        text,
                    })
                }
            }
        }
        None
    }

    /// The toolbar of the edited mode: *Commit*, greyed with the refusal the
    /// step would give; *Discard* and *Rename*, greyed only while a task holds
    /// the node, since neither edits the item.
    fn show_toolbar(
        &mut self,
        ui: &mut egui::Ui,
        state: &AppState,
        id: ReconId,
        label: &str,
        response: &mut MeasurementViewResponse,
    ) {
        let busy = state.busy_refusal(id);
        let commit_refusal = state.measurement_commit_refusal(id, label);
        ui.horizontal_wrapped(|ui| {
            self.toolbar
                .push(("Commit".to_string(), commit_refusal.clone()));
            if entry(
                ui,
                "Commit",
                commit_refusal,
                "Write this measurement into the reconstruction",
            ) {
                response.commit = Some((id, label.to_string()));
            }
            self.toolbar.push(("Discard".to_string(), busy.clone()));
            if entry(
                ui,
                "Discard",
                busy.clone(),
                "Take this measurement off the bench",
            ) {
                response.discard = Some((id, label.to_string()));
            }
            self.toolbar.push(("Rename".to_string(), busy.clone()));
            match self.renaming.as_mut().filter(|(item, _)| item == label) {
                None => {
                    if entry(ui, "Rename", busy, "Give this item a name of your own") {
                        self.renaming = Some((label.to_string(), label.to_string()));
                    }
                }
                Some((item, text)) => {
                    let item = item.clone();
                    let field = ui.add(egui::TextEdit::singleline(text).desired_width(140.0));
                    crate::widget_name::set_widget_name(&field, RENAME_NAME);
                    let commit =
                        field.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
                    let typed = text.clone();
                    if commit || ui.button("Set").clicked() {
                        response.rename = Some((id, item, typed));
                        self.renaming = None;
                    } else if ui.button("Cancel").clicked() {
                        self.renaming = None;
                    }
                }
            }
        });
    }

    /// The two endpoint blocks side by side, each with its pictures cut on
    /// the first frame that shows it.
    fn show_blocks(
        &mut self,
        ui: &mut egui::Ui,
        state: &AppState,
        node: &SceneNode,
        target: Target,
        blocks: &[Option<Endpoint>; 2],
        response: &mut MeasurementViewResponse,
    ) {
        let document = node.history.current_version().document_serial;
        let unit = node.recon().metadata.world_space_unit.as_deref();
        let keys: Vec<PictureKey> = EndpointSide::BOTH
            .into_iter()
            .filter_map(|side| {
                let endpoint = blocks[side.index()].as_ref()?;
                Some(PictureKey::new(
                    node.id,
                    document,
                    side,
                    endpoint.image,
                    endpoint.pixel,
                ))
            })
            .collect();
        self.pictures.retain(|key, _| keys.contains(key));
        ui.columns(2, |columns| {
            for side in EndpointSide::BOTH {
                let ui = &mut columns[side.index()];
                let Some(endpoint) = &blocks[side.index()] else {
                    show_unplaced(ui, side);
                    continue;
                };
                let key = PictureKey::new(node.id, document, side, endpoint.image, endpoint.pixel);
                let block = EndpointBlock {
                    node,
                    target,
                    image: endpoint.image,
                    pixel: endpoint.pixel,
                    side,
                    resolution: &endpoint.resolution,
                    selected: state.selected_endpoint == Some(side),
                    unit,
                };
                self.ensure_pictures(ui.ctx(), state, &block, key);
                let pictures = self.pictures.get(&key);
                self.names[side.index()] = block.show(ui, pictures, response);
            }
        });
    }

    /// No measurement selected: one line saying so, then, where the
    /// reconstruction has measurements, each as a link that selects it, so
    /// the panel is usable without the Scene tree.
    fn show_empty_state(
        &mut self,
        ui: &mut egui::Ui,
        node: &SceneNode,
        rows: &[DistanceRow],
        resolved: &[ResolvedDistanceMeasurement],
        response: &mut MeasurementViewResponse,
    ) {
        ui.label(NO_MEASUREMENT);
        if rows.is_empty() {
            ui.weak(format!("{} stores no measurements.", node.label));
            return;
        }
        ui.add_space(4.0);
        ui.weak(format!("The measurements {} stores:", node.label));
        let unit = node.recon().metadata.world_space_unit.as_deref();
        for (index, row) in rows.iter().enumerate() {
            let Some(reading) = resolved.get(index) else {
                continue;
            };
            let measurement = MeasurementRef::distance(node.id, index);
            let link = ui.link(measurements::row_text(measurement, row, reading, unit));
            self.links.push(link.rect);
            if link.clicked() {
                response.select_measurement = Some(measurement);
            }
        }
    }

    /// Cut the pictures of the endpoint `block` shows when they are not held
    /// for `key` yet: its crop once its photograph is decoded, and its point's
    /// stored patch.
    ///
    /// A photograph still decoding leaves nothing held, so the next frame asks
    /// again; [`crate::state::display_photograph`] repaints when it lands.
    fn ensure_pictures(
        &mut self,
        ctx: &egui::Context,
        state: &AppState,
        block: &EndpointBlock<'_>,
        key: PictureKey,
    ) {
        if self.pictures.contains_key(&key) {
            return;
        }
        let Some(crop) = endpoint_crop(ctx, state, block, &self.regions) else {
            return;
        };
        let patch = block.resolution.as_ref().ok().and_then(|resolution| {
            let view = block.node.edited().point(resolution.chosen().point)?;
            let image = crate::patch_pictures::stored_patch_image(view.patch_bitmap()?)?;
            Some(ctx.load_texture(
                format!(
                    "measurement_patch_{:?}_{}_{}",
                    key.node,
                    block.texture_stem(),
                    block.side.letter()
                ),
                image,
                egui::TextureOptions::NEAREST,
            ))
        });
        self.pictures.insert(key, EndpointPictures { crop, patch });
    }
}

/// Which measurement an endpoint block is a block of, which says where its
/// gestures go.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Target {
    /// A stored row: the read-only mode.
    Stored(MeasurementRef),
    /// The focused item on this node's bench: the edited mode.
    Item(ReconId),
}

/// One placed endpoint, with what the version makes of it.
struct Endpoint {
    image: u32,
    pixel: [f64; 2],
    resolution: Result<EndpointResolution, EndpointRefusal>,
}

/// The *Edit* box, Track View's, with the same meaning: ticked exactly when
/// the focused item is a measurement on `id`'s bench.
///
/// A tick puts the selected measurement on the bench and focuses it, or
/// focuses its item where it is on the bench already; with no measurement
/// selected it focuses the most recent measurement item, and with none of
/// those it is greyed naming the ways in. Only a tick that would put a
/// measurement on is greyed by a task holding the node, since focusing and
/// unfocusing push no version.
fn show_edit_box(
    ui: &mut egui::Ui,
    state: &AppState,
    id: ReconId,
    editing: bool,
    selected: Option<MeasurementRef>,
    response: &mut MeasurementViewResponse,
) {
    let recent = state.most_recent_item_of(sfmtool_core::bench::ItemKind::Measurement);
    let refusal = if editing {
        None
    } else {
        match selected {
            None => recent
                .is_none()
                .then(crate::bench::measurement::nothing_to_edit),
            Some(measurement) => {
                if state.bench_item_from_measurement(measurement).is_some() {
                    None
                } else {
                    state.busy_refusal(id)
                }
            }
        }
    };
    let hint = match (editing, selected, &recent) {
        (true, _, _) => "Stop editing this measurement: it stays on the bench, and the panel \
                         shows the stored measurement it came from, when it has one"
            .to_string(),
        (false, None, Some((_, label))) => format!("Edit {label} again"),
        (false, _, _) => "Put the selected measurement on the bench and edit it".to_string(),
    };
    let mut checked = editing;
    let checkbox = ui.add_enabled(
        refusal.is_none(),
        egui::Checkbox::new(&mut checked, EDIT_LABEL),
    );
    let checkbox = match &refusal {
        Some(why) => checkbox.on_disabled_hover_text(why),
        None => checkbox.on_hover_text(hint),
    };
    if checkbox.changed() {
        response.set_edit = Some((id, checked));
    }
}

/// A toolbar button: enabled with `hint` as its hover text, or greyed with
/// the refusal as its hover text. Whether it was clicked.
fn entry(ui: &mut egui::Ui, text: &str, refusal: Option<String>, hint: &str) -> bool {
    let button = egui::Button::new(text);
    match refusal {
        None => ui.add(button).on_hover_text(hint).clicked(),
        Some(why) => {
            ui.add_enabled(false, button).on_disabled_hover_text(why);
            false
        }
    }
}

/// An endpoint that is not placed: a block saying so.
fn show_unplaced(ui: &mut egui::Ui, side: EndpointSide) {
    egui::Frame::group(ui.style()).show(ui, |ui| {
        ui.set_width(ui.available_width());
        ui.label(egui::RichText::new(side.letter()).strong());
        ui.weak(format!("Endpoint {} is not placed.", side.letter()));
    });
}

/// The result block's line for an item that has no result: what it lacks.
pub(crate) fn item_without_result(item: &EditableMeasurement) -> String {
    let Some(distance) = item.as_distance() else {
        return "No distance.".to_string();
    };
    for side in EndpointSide::BOTH {
        if distance.endpoints[side.index()].is_none() {
            return format!("No distance: endpoint {} is not placed.", side.letter());
        }
    }
    "No distance: no stated distance is given.".to_string()
}

/// The start of the edited mode's header: the item's bench label and node,
/// and where it came from. The label and stated distance fields follow
/// ([`MeasurementView::show_header_fields`]), then the median line
/// ([`show_item_median`]).
fn show_item_origin(
    ui: &mut egui::Ui,
    node: &SceneNode,
    label: &str,
    item: &EditableMeasurement,
    origin: Option<u32>,
) {
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new(label).strong());
        ui.weak("on the bench of");
        ui.label(&node.label);
    });
    match origin {
        Some(index) => {
            let from = MeasurementRef {
                recon: node.id,
                kind: item.kind(),
                index,
            };
            ui.weak(format!("From {}: Commit replaces it", from.title()));
        }
        None => {
            ui.weak("From no stored row: Commit adds it as a new row");
        }
    }
}

/// The end of the edited mode's header: the median line, which compares the
/// item with the stored measurements other than its origin.
fn show_item_median(
    ui: &mut egui::Ui,
    row: Option<&DistanceRow>,
    comparison: impl FnOnce() -> Option<measurements::ScaleComparison>,
) {
    if let (Some(row), Some(line)) = (row, comparison()) {
        ui.label(measurements::scale_comparison_text(&line, row.unit))
            .on_hover_text(
                "The median of the scales this measurement and the other stored ones imply, \
                 and how far this one's is from it. su: one unit of the reconstruction's own \
                 coordinates.",
            );
    }
}

/// The header: the kind, index and reconstruction, the label, the stated
/// distance, and the line comparing this measurement with the others of its
/// kind.
fn show_header(
    ui: &mut egui::Ui,
    node: &SceneNode,
    measurement: MeasurementRef,
    row: &DistanceRow,
    rows: &[DistanceRow],
    resolved: &[ResolvedDistanceMeasurement],
) {
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new(measurement.title()).strong());
        ui.weak("in");
        ui.label(&node.label);
    });
    if row.label.is_empty() {
        ui.weak("No label");
    } else {
        ui.label(format!("Label: {}", row.label));
    }
    ui.label(format!("Stated distance: {}", row.stated_text()));
    if let Some(line) = measurements::scale_comparison(rows, resolved, measurement.index()) {
        ui.label(measurements::scale_comparison_text(&line, row.unit))
            .on_hover_text(
                "The median of the scales the measurements with a reading imply, and how far \
             this one's is from it. su: one unit of the reconstruction's own coordinates.",
            );
    }
}

/// The result block: exactly one of the four results.
fn show_result(
    ui: &mut egui::Ui,
    row: &DistanceRow,
    resolved: &ResolvedDistanceMeasurement,
    unit: Option<&str>,
) {
    ui.label(egui::RichText::new("Result").strong());
    for line in result_lines(row, resolved, unit) {
        match line {
            ResultLine::Text(text) => {
                ui.label(text);
            }
            ResultLine::Hovered(text, hover) => {
                ui.label(text).on_hover_text(hover);
            }
        }
    }
}

/// One line of the result block, with the hover text it carries.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum ResultLine {
    Text(String),
    Hovered(String, String),
}

impl ResultLine {
    /// The line's text.
    #[cfg(test)]
    pub(crate) fn text(&self) -> &str {
        match self {
            ResultLine::Text(text) | ResultLine::Hovered(text, _) => text,
        }
    }
}

/// The lines of the result block.
pub(crate) fn result_lines(
    row: &DistanceRow,
    resolved: &ResolvedDistanceMeasurement,
    world_space_unit: Option<&str>,
) -> Vec<ResultLine> {
    let unit = row.unit.name();
    match measurements::distance_result(row, resolved, world_space_unit) {
        DistanceResult::WithUnit {
            reconstructed,
            error,
            error_percent,
            implied_scale,
        } => {
            let words = if error >= 0.0 {
                format!(
                    "the reconstruction is {} {unit} longer than the stated distance",
                    measurements::significant(error.abs(), 3)
                )
            } else {
                format!(
                    "the reconstruction is {} {unit} shorter than the stated distance",
                    measurements::significant(error.abs(), 3)
                )
            };
            vec![
                ResultLine::Text(format!(
                    "Reconstructed: {} {unit}",
                    measurements::significant(reconstructed, 4)
                )),
                ResultLine::Hovered(
                    format!(
                        "Error: {} {unit} ({}%)",
                        signed_significant(error, 3),
                        measurements::signed(error_percent, 2)
                    ),
                    format!("Reconstructed minus stated: {words}"),
                ),
                ResultLine::Hovered(
                    format!(
                        "Implied scale: {}{}",
                        measurements::significant(implied_scale, 5),
                        "\u{00d7}"
                    ),
                    "Stated over reconstructed: the factor the reconstruction's lengths \
                     would be scaled by to agree with this measurement"
                        .to_string(),
                ),
            ]
        }
        DistanceResult::SceneUnits {
            scene_distance,
            unit_per_scene_unit,
        } => vec![
            ResultLine::Hovered(
                format!(
                    "Reconstructed: {} su",
                    measurements::significant(scene_distance, 4)
                ),
                "su: the reconstruction declares no unit, so its lengths are in scene units"
                    .to_string(),
            ),
            ResultLine::Hovered(
                format!(
                    "Implied scale: 1 su = {} {unit}",
                    measurements::significant(unit_per_scene_unit, 4)
                ),
                "What sfm xform --scale-by-measurements would apply from this measurement \
                 alone"
                    .to_string(),
            ),
        ],
        DistanceResult::AtInfinity(side) => vec![ResultLine::Text(format!(
            "No distance: endpoint {} lies on a point at infinity.",
            side.letter()
        ))],
        DistanceResult::Refused(side) => vec![ResultLine::Text(format!(
            "No distance: endpoint {} has no 3D position.",
            side.letter()
        ))],
    }
}

/// `value` with `digits` significant digits and its sign, the minus as
/// [`MINUS`].
fn signed_significant(value: f64, digits: i32) -> String {
    let text = measurements::significant(value.abs(), digits);
    if value < 0.0 {
        format!("{MINUS}{text}")
    } else {
        format!("+{text}")
    }
}

/// One endpoint block's inputs.
struct EndpointBlock<'a> {
    node: &'a SceneNode,
    target: Target,
    image: u32,
    pixel: [f64; 2],
    side: EndpointSide,
    resolution: &'a Result<EndpointResolution, EndpointRefusal>,
    selected: bool,
    unit: Option<&'a str>,
}

impl EndpointBlock<'_> {
    /// What the block's textures are named by: the endpoint's image and
    /// pixel, which is all its pictures are cut from.
    fn texture_stem(&self) -> String {
        format!("{}_{:.3}_{:.3}", self.image, self.pixel[0], self.pixel[1])
    }

    /// Draw the block and report its gestures into `response`, returning
    /// where its name line was drawn.
    fn show(
        &self,
        ui: &mut egui::Ui,
        pictures: Option<&EndpointPictures>,
        response: &mut MeasurementViewResponse,
    ) -> Option<Rect> {
        let visuals = ui.visuals().clone();
        let frame = if self.selected {
            egui::Frame::group(ui.style())
                .fill(visuals.selection.bg_fill.gamma_multiply(0.35))
                .stroke(visuals.selection.stroke)
        } else {
            egui::Frame::group(ui.style())
        };
        let recon = self.node.recon();
        let image = self.image;
        let [x, y] = self.pixel;
        let mut targets: Vec<egui::Response> = Vec::new();
        let mut name_rect = None;
        frame.show(ui, |ui| {
            ui.set_width(ui.available_width());
            let name = ui.add(
                egui::Label::new(
                    egui::RichText::new(format!(
                        "{}  {}",
                        self.side.letter(),
                        measurements::image_name(recon, image)
                    ))
                    .strong(),
                )
                .sense(egui::Sense::click())
                .truncate(),
            );
            name_rect = Some(name.rect);
            targets.push(name);
            targets.push(
                ui.add(egui::Label::new(format!("({x:.1}, {y:.1})")).sense(egui::Sense::click())),
            );
            ui.horizontal(|ui| {
                targets.push(self.show_crop(ui, pictures));
                if let Some(patch) = pictures.and_then(|p| p.patch.as_ref()) {
                    targets.push(self.show_patch(ui, patch));
                }
            });
            self.show_resolution(ui, response);
        });
        let hovered = targets.iter().any(egui::Response::hovered);
        if hovered {
            response.hovered_image = Some(ImageRef::new(self.node.id, image as usize));
        }
        if targets.iter().any(egui::Response::clicked) {
            match self.target {
                Target::Stored(measurement) => {
                    response.select_endpoint = Some((measurement, self.side))
                }
                Target::Item(id) => response.select_item_endpoint = Some((id, self.side)),
            }
        }
        if targets.iter().any(egui::Response::double_clicked) {
            match self.target {
                Target::Stored(measurement) => {
                    response.look_through = Some((measurement, self.side))
                }
                Target::Item(id) => response.look_through_item = Some((id, self.side)),
            }
        }
        name_rect
    }

    /// The crop cell, with its hover view.
    fn show_crop(&self, ui: &mut egui::Ui, pictures: Option<&EndpointPictures>) -> egui::Response {
        let (rect, response) =
            ui.allocate_exact_size(egui::vec2(CROP_SIDE, CROP_SIDE), egui::Sense::click());
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 2.0, ui.visuals().extreme_bg_color);
        let full_name = measurements::image_name(self.node.recon(), self.image);
        match pictures.map(|p| &p.crop) {
            Some(Crop::Ready {
                cell,
                context,
                caption,
            }) => {
                cell.paint_cell(&painter, rect);
                paint_cross(&painter, cell.keypoint_in(rect));
                let context = context.as_ref();
                response.on_hover_ui(|ui| {
                    ui.label(egui::RichText::new(&full_name).strong());
                    if let Some(context) = context {
                        let bounds = Rect::from_min_size(
                            Pos2::ZERO,
                            egui::vec2(CONTEXT_HOVER_SIDE, CONTEXT_HOVER_SIDE),
                        );
                        let (view, _) = ui.allocate_exact_size(
                            context.fitted(bounds).size(),
                            egui::Sense::hover(),
                        );
                        let painter = ui.painter_at(view);
                        let halo = Stroke::new(3.0, Color32::from_black_alpha(170));
                        context.paint_cell_with(
                            &painter,
                            view,
                            Color32::WHITE,
                            &[halo, Stroke::new(1.5, outline_color())],
                        );
                        paint_cross(&painter, context.keypoint_in(view));
                    }
                    ui.set_max_width(CONTEXT_HOVER_SIDE);
                    ui.label(caption);
                })
            }
            Some(Crop::Absent(why)) => {
                painter.text(
                    rect.center(),
                    egui::Align2::CENTER_CENTER,
                    "no crop",
                    egui::FontId::proportional(11.0),
                    ui.visuals().weak_text_color(),
                );
                response.on_hover_text(format!("{full_name}\n{why}"))
            }
            None => {
                painter.text(
                    rect.center(),
                    egui::Align2::CENTER_CENTER,
                    "loading…",
                    egui::FontId::proportional(11.0),
                    ui.visuals().weak_text_color(),
                );
                response.on_hover_text(format!("{full_name}\nThe photograph is being read"))
            }
        }
    }

    /// The chosen point's stored patch, with a cross at the endpoint's texel.
    fn show_patch(&self, ui: &mut egui::Ui, patch: &egui::TextureHandle) -> egui::Response {
        let (rect, response) =
            ui.allocate_exact_size(egui::vec2(PATCH_SIDE, PATCH_SIDE), egui::Sense::click());
        let painter = ui.painter_at(rect);
        painter.image(
            patch.id(),
            rect,
            Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)),
            Color32::WHITE,
        );
        if let Ok(resolution) = self.resolution {
            painter_cross_at_texel(&painter, rect, resolution.chosen().patch_coords);
        }
        response.on_hover_text(
            "The point's stored patch, with a cross at the place on it the endpoint names",
        )
    }

    /// How the endpoint was resolved: the point, the texel and the position,
    /// with the candidates where it was ranked; or why it has no position.
    fn show_resolution(&self, ui: &mut egui::Ui, response: &mut MeasurementViewResponse) {
        let resolution = match self.resolution {
            Ok(resolution) => resolution,
            Err(refusal) => {
                ui.colored_label(
                    ui.visuals().warn_fg_color,
                    format!("No position: {refusal}."),
                );
                return;
            }
        };
        let chosen = resolution.chosen();
        let id = crate::scene::point_id(self.node, chosen.point as usize);
        ui.horizontal(|ui| {
            ui.label(egui::RichText::new(&id).monospace());
            if crate::track_view::header_buttons::copy_button(ui, "Copy Point ID") {
                ui.ctx().copy_text(id.clone());
            }
            if crate::track_view::header_buttons::goto_button(ui) {
                response.goto_point = Some(id.clone());
            }
        });
        let [s, t] = chosen.patch_coords;
        ui.label(format!("Texel (s, t): ({s:.3}, {t:.3})"));
        match chosen.position {
            EndpointPosition::Finite { position } => {
                ui.label(format!("Position: {}", position_text(position, self.unit)));
            }
            EndpointPosition::AtInfinity { direction } => {
                ui.label(format!(
                    "Direction: ({:.4}, {:.4}, {:.4})",
                    direction[0], direction[1], direction[2]
                ));
                ui.label(
                    "This endpoint lies on a point at infinity, so no distance can be \
                     measured to it.",
                );
            }
        }
        match chosen.standing {
            CandidateStanding::Alone => {
                ui.weak("One square holds the pixel.");
            }
            _ => {
                egui::CollapsingHeader::new(format!(
                    "Chosen from {} squares",
                    resolution.candidates.len()
                ))
                .id_salt(("measurement_candidates", self.side.index()))
                .show(ui, |ui| {
                    for (i, candidate) in resolution.candidates.iter().enumerate() {
                        let id = crate::scene::point_id(self.node, candidate.point as usize);
                        let chosen = if i == resolution.chosen {
                            ", chosen"
                        } else {
                            ""
                        };
                        ui.label(format!(
                            "{id}  {}{chosen}",
                            standing_text(candidate.standing)
                        ));
                    }
                });
            }
        }
    }
}

/// What a candidate's standing reads as in the candidates list: its cost and
/// its costliest term, or why it was left out.
pub(crate) fn standing_text(standing: CandidateStanding) -> String {
    match standing {
        CandidateStanding::Alone => "alone".to_string(),
        CandidateStanding::Ranked(cost) => {
            let costliest = cost
                .ln_parts()
                .into_iter()
                .max_by(|a, b| a.1.total_cmp(&b.1))
                .map_or("none", |(term, _)| term.name());
            format!(
                "cost {}, costliest term {costliest}",
                measurements::significant(cost.cost, 3)
            )
        }
        CandidateStanding::Excluded(why) => format!("excluded: {}", why.name().replace('_', " ")),
    }
}

/// A position in the file's unit, or in scene units where it declares none.
fn position_text(position: [f64; 3], unit: Option<&str>) -> String {
    let unit = unit
        .filter(|u| LengthUnit::from_name(u).is_some())
        .unwrap_or("su");
    format!(
        "({}, {}, {}) {unit}",
        measurements::significant(position[0], 4),
        measurements::significant(position[1], 4),
        measurements::significant(position[2], 4)
    )
}

/// The colour a crop's square is outlined in.
fn outline_color() -> Color32 {
    Color32::from_rgb(120, 200, 255)
}

/// A cross at `at`: two light strokes over wider dark ones, so it reads over
/// a bright photograph and a dark one alike.
fn paint_cross(painter: &egui::Painter, at: Pos2) {
    let arm = 6.0;
    let halo = Stroke::new(3.0, Color32::from_black_alpha(190));
    let light = Stroke::new(1.5, Color32::from_rgb(255, 80, 80));
    for stroke in [halo, light] {
        painter.line_segment(
            [at - egui::vec2(arm, 0.0), at + egui::vec2(arm, 0.0)],
            stroke,
        );
        painter.line_segment(
            [at - egui::vec2(0.0, arm), at + egui::vec2(0.0, arm)],
            stroke,
        );
    }
}

/// A cross at texel `(s, t)` of a patch drawn over `rect`: `s` runs left to
/// right and `t` bottom to top, since a patch bitmap's rows step along `−v`.
fn painter_cross_at_texel(painter: &egui::Painter, rect: Rect, [s, t]: [f64; 2]) {
    let x = rect.left() + rect.width() * ((s as f32 + 1.0) / 2.0);
    let y = rect.top() + rect.height() * ((1.0 - t as f32) / 2.0);
    paint_cross(painter, Pos2::new(x, y));
}

/// The crop of the endpoint `block` shows, or `None` while its photograph is
/// decoding.
///
/// Where the endpoint resolves, the crop is cut around the chosen candidate's
/// square in that image, as Track View's crop is cut around a row's square.
/// Where it does not, it is a [`UNRESOLVED_CROP_PX`] square of the photograph
/// centred on the pixel, with no outline.
fn endpoint_crop(
    ctx: &egui::Context,
    state: &AppState,
    block: &EndpointBlock<'_>,
    regions: &PanelRegionPool,
) -> Option<Crop> {
    let recon = block.node.recon();
    let image = block.image as usize;
    if image >= recon.image_set.images.len() {
        return Some(Crop::Absent("The image is not in the reconstruction"));
    }
    let pixel = block.pixel;
    let square = block.resolution.as_ref().ok().and_then(|resolution| {
        let chosen = resolution.chosen();
        let view = block.node.edited().point(chosen.point)?;
        let patch_frame = view.patch_frame()?;
        let keypoint = view.keypoint_xy(chosen.observation)?;
        let (camera, pose) = crate::bench::geometry::view_of(&recon.image_set, image)?;
        let mut outline = crop::patch_outline(
            &patch_frame,
            &camera,
            &pose,
            [f64::from(keypoint[0]), f64::from(keypoint[1])],
            view.point().position,
            crate::patch_pictures::marks::ProjectionOf::Point,
        );
        // The crop marks the endpoint's pixel, not the observation's keypoint
        // or the point's projection.
        outline.keypoint = pixel;
        outline.projection = None;
        outline.projection_of = None;
        Some(outline)
    });
    let resolved = square.is_some();
    let outline = square.unwrap_or_else(|| square_around(pixel));
    let pyramid = match crate::state::display_photograph(
        &state.photographs,
        &state.photograph_requests,
        recon,
        image,
        ctx,
    ) {
        DisplayPhotograph::Decoded(pyramid) => pyramid,
        DisplayPhotograph::Decoding => return None,
        DisplayPhotograph::Unreadable => {
            return Some(Crop::Absent("The photograph could not be read"))
        }
    };
    let source = TiledPhotograph::from_pyramid(pyramid);
    let pool = regions.pool();
    let cut = |context: bool| {
        let picture = if context {
            crop::context(&outline, &source, pool)
        } else {
            crop::image(&outline, &source, pool)
        };
        picture.ok().flatten().map(|mut picture| {
            if !resolved {
                picture.outline.clear();
            }
            picture
        })
    };
    let Some(picture) = cut(false) else {
        return Some(Crop::Absent(
            "The endpoint's place does not land in the photograph",
        ));
    };
    let [w, h] = picture.crop.size;
    let axes = |axis: Option<f64>| {
        axis.map_or("not all in this photograph".to_string(), |px| {
            format!("{px:.1} px")
        })
    };
    let caption = if resolved {
        format!(
            "\u{2022} Pixel: ({:.1}, {:.1}), at the cross\n\
             \u{2022} Crop: {w} x {h} px of the photograph\n\
             \u{2022} Square: {} across, {} up",
            pixel[0],
            pixel[1],
            axes(picture.axes_px[0]),
            axes(picture.axes_px[1]),
        )
    } else {
        format!(
            "\u{2022} Pixel: ({:.1}, {:.1}), at the cross\n\
             \u{2022} Crop: {w} x {h} px of the photograph around it\n\
             \u{2022} No square holds the pixel",
            pixel[0], pixel[1],
        )
    };
    let name = format!(
        "measurement_crop_{:?}_{}_{}",
        block.node.id,
        block.texture_stem(),
        block.side.letter()
    );
    let cell = Box::new(DrawnCrop::new(
        ctx,
        picture,
        outline_color(),
        name.clone(),
        CELL_TEXTURE_OPTIONS,
    ));
    let context = cut(true).map(|picture| {
        Box::new(DrawnCrop::new(
            ctx,
            picture,
            outline_color(),
            format!("{name}_context"),
            egui::TextureOptions::LINEAR,
        ))
    });
    Some(Crop::Ready {
        cell,
        context,
        caption,
    })
}

/// A square of [`UNRESOLVED_CROP_PX`] photograph pixels centred on `pixel`, as
/// an outline the crop is cut around.
fn square_around(pixel: [f64; 2]) -> Outline {
    let half = UNRESOLVED_CROP_PX / 2.0;
    let [x, y] = pixel;
    Outline {
        samples: vec![
            Some([x - half, y - half]),
            Some([x + half, y - half]),
            Some([x + half, y + half]),
            Some([x - half, y + half]),
        ],
        axes_px: [None, None],
        keypoint: pixel,
        projection: None,
        projection_of: None,
    }
}
