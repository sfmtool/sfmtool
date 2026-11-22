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
//! buttons keep their clicks. In the edited mode each block's image name is a
//! drop-down of the reconstruction's images and its pixel two number fields,
//! each of which sets the endpoint's image and pixel in one step, and the block
//! also has *Switch image…*, and a right-click menu holding *Switch Image…*, which open a
//! popup of the other photographs that show the endpoint's place (core's
//! `endpoint_views`); a click on one of its rows asks for the step that moves
//! the endpoint there. The blocks stand side by side where the panel is wide
//! enough for two, and one above the other where it is not.
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
    endpoint_views, CandidateStanding, EndpointPosition, EndpointRefusal, EndpointResolution,
    EndpointView, MeasurementEndpointOptions, ResolvedDistanceMeasurement,
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
    /// A row of an endpoint's *Switch image…* popup was clicked.
    pub(crate) switch_endpoint: Option<EndpointSwitch>,
    /// An endpoint block's image drop-down or pixel field was committed: the
    /// endpoint set to that image and pixel.
    pub(crate) set_endpoint: Option<crate::bench::measurement::EndpointSet>,
    /// A pixel field was committed with text that is not a number: the
    /// sentence the dock writes as a failed row.
    pub(crate) endpoint_refused: Option<String>,
}

/// A row of the *Switch image…* popup that was clicked: move endpoint `side`
/// of the item `item` on `node`'s bench to `pixel` of `image`.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct EndpointSwitch {
    pub(crate) node: ReconId,
    pub(crate) item: String,
    pub(crate) side: EndpointSide,
    pub(crate) image: u32,
    pub(crate) pixel: [f64; 2],
}

/// The text of each endpoint block's button that opens the popup.
pub(crate) const SWITCH_LABEL: &str = "Switch image\u{2026}";

/// The entry of an endpoint block's right-click menu that opens the popup.
pub(crate) const SWITCH_ENTRY: &str = "Switch Image\u{2026}";

/// The side of a crop in the *Switch image…* popup, in points.
const OFFER_CROP_SIDE: f32 = 96.0;

/// The narrowest an endpoint block is drawn, in points: its crop and its
/// patch side by side, with the frame's margins. Narrower than two of these,
/// the panel stacks the blocks rather than putting them side by side.
const BLOCK_MIN_WIDTH: f32 = CROP_SIDE + PATCH_SIDE + 40.0;

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

/// A field of the edited mode being typed in: which item, the text the field
/// held when it took focus, and the text typed so far.
struct FieldDraft {
    item: String,
    original: String,
    text: String,
}

/// One field of the edited mode, as [`draft_field`] draws it.
struct DraftField<'a> {
    /// The item the field edits, by its label on the bench.
    item: &'a str,
    /// What the field holds while nobody is typing in it.
    value: &'a str,
    /// Why a step on the item would be refused, which greys the field.
    refusal: Option<&'a str>,
    /// The grey text an empty field shows.
    empty: &'a str,
    /// The hover text of a field that can be typed in.
    hint: &'a str,
    width: f32,
    /// The field's name in the widget listing.
    name: &'a str,
}

/// A text field of the edited mode, holding `field.value` while nobody is
/// typing in it and its draft in `slot` while somebody is. What it was set
/// to, on the frame it is committed: Enter, or the field losing focus, with
/// text other than the text it held when it took focus. Escape puts the field
/// back and commits nothing. Where it was drawn comes back too.
///
/// Each field keeps its own draft, so moving the focus straight from one
/// field to another commits the first once rather than losing it or handing
/// its text to the second.
fn draft_field(
    ui: &mut egui::Ui,
    slot: &mut Option<FieldDraft>,
    field: DraftField<'_>,
) -> (Option<String>, Rect) {
    if slot.as_ref().is_some_and(|draft| draft.item != field.item) {
        *slot = None;
    }
    let mut text = match slot {
        Some(draft) => draft.text.clone(),
        None => field.value.to_string(),
    };
    let edit = egui::TextEdit::singleline(&mut text)
        .desired_width(field.width)
        .hint_text(field.empty);
    let shown = ui.add_enabled(field.refusal.is_none(), edit);
    crate::widget_name::set_widget_name(&shown, field.name);
    let rect = shown.rect;
    let shown = match field.refusal {
        Some(why) => shown.on_disabled_hover_text(why),
        None => shown.on_hover_text(field.hint),
    };
    if shown.lost_focus() {
        let cancelled = ui.input(|i| i.key_pressed(egui::Key::Escape));
        let original = slot
            .take()
            .map_or_else(|| field.value.to_string(), |draft| draft.original);
        return ((!cancelled && text != original).then_some(text), rect);
    }
    if shown.has_focus() {
        match slot {
            Some(draft) => draft.text = text,
            slot => {
                *slot = Some(FieldDraft {
                    item: field.item.to_string(),
                    original: field.value.to_string(),
                    text,
                })
            }
        }
    } else {
        // Not focused, so whatever was typed here was committed or put back
        // on the frame focus left.
        *slot = None;
    }
    (None, rect)
}

/// One of an endpoint block's two pixel fields.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PixelAxis {
    X,
    Y,
}

impl PixelAxis {
    const BOTH: [PixelAxis; 2] = [PixelAxis::X, PixelAxis::Y];

    fn index(self) -> usize {
        match self {
            PixelAxis::X => 0,
            PixelAxis::Y => 1,
        }
    }

    fn letter(self) -> &'static str {
        match self {
            PixelAxis::X => "x",
            PixelAxis::Y => "y",
        }
    }
}

/// The name of endpoint `side`'s image drop-down in the widget listing.
pub(crate) fn image_field_name(side: EndpointSide) -> &'static str {
    match side {
        EndpointSide::A => "endpoint A image",
        EndpointSide::B => "endpoint B image",
    }
}

/// The name of endpoint `side`'s pixel field `axis` in the widget listing.
pub(crate) fn pixel_field_name(side: EndpointSide, axis: PixelAxis) -> &'static str {
    match (side, axis) {
        (EndpointSide::A, PixelAxis::X) => "endpoint A pixel x",
        (EndpointSide::A, PixelAxis::Y) => "endpoint A pixel y",
        (EndpointSide::B, PixelAxis::X) => "endpoint B pixel x",
        (EndpointSide::B, PixelAxis::Y) => "endpoint B pixel y",
    }
}

/// What a pixel field shows of a stored coordinate: the shortest text that
/// reads back as the same `float32`, which is how the file stores it, so a
/// field left as it is commits nothing and one typed over holds what was
/// typed.
pub(crate) fn pixel_text(coordinate: f64) -> String {
    format!("{}", coordinate as f32)
}

/// What an endpoint block's fields keep between frames: each pixel field's
/// draft, and where the fields and the drop-down's rows were drawn on the
/// last frame, for the tests that aim at them.
#[derive(Default)]
struct EndpointFields {
    drafts: [Option<FieldDraft>; 2],
    pixel_rects: [Option<Rect>; 2],
    image_rect: Option<Rect>,
    /// The drop-down's rows while it is open: where each was drawn, and the
    /// image it names.
    choices: Vec<(Rect, u32)>,
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
    /// Each header field's typing in progress, indexed by
    /// [`HeaderField::index`]. One per field, so moving the focus straight from
    /// one field to the other commits the first rather than losing it.
    drafts: [Option<FieldDraft>; 2],
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
    /// Where each endpoint block was drawn on the last frame, its frame and
    /// everything in it.
    blocks: [Option<Rect>; 2],
    /// Each endpoint block's *Switch image…* button as the last frame drew
    /// it, with the refusal it was greyed with, if any.
    switches: [Option<(Rect, Option<String>)>; 2],
    /// Where each block's right-click menu drew *Switch Image…* on the last
    /// frame, while the menu is open.
    menu_entries: [Option<Rect>; 2],
    /// The offers of the endpoint whose popup is open.
    offers: Option<HeldOffers>,
    /// The crops of the popup's rows, cut once each photograph is decoded.
    offer_crops: HashMap<OfferKey, Crop>,
    /// The popup's rows as the last frame drew them: where each row's image
    /// name was drawn, and the image and pixel the row moves the endpoint to.
    offer_rows: Vec<(Rect, u32, [f64; 2])>,
    /// Each endpoint block's image drop-down and pixel fields in the edited
    /// mode, indexed by [`EndpointSide::index`].
    endpoint_fields: [EndpointFields; 2],
}

/// The offers of one endpoint of the focused item, read once per item value,
/// document and side ([`sfmtool_core::reconstruction::measurement_endpoint::endpoint_views`]).
struct HeldOffers {
    item: Arc<EditableMeasurement>,
    document: VersionSerial,
    side: EndpointSide,
    views: Result<Vec<EndpointView>, EndpointRefusal>,
}

/// Which row's crop a cached offer crop is: the observation it is cut
/// around and the pixel it marks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct OfferKey {
    node: ReconId,
    document: VersionSerial,
    point: u32,
    observation: usize,
    pixel: [u64; 2],
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

    /// Where the endpoint block on `side` was drawn on the last frame.
    #[cfg(test)]
    pub(crate) fn block_rect(&self, side: EndpointSide) -> Option<Rect> {
        self.blocks[side.index()]
    }

    /// The *Switch image…* button of the block on `side` as the last frame
    /// drew it: where, and the refusal it was greyed with.
    #[cfg(test)]
    pub(crate) fn switch_button(&self, side: EndpointSide) -> Option<&(Rect, Option<String>)> {
        self.switches[side.index()].as_ref()
    }

    /// Where the right-click menu of the block on `side` drew *Switch
    /// Image…* on the last frame, while the menu is open.
    #[cfg(test)]
    pub(crate) fn menu_entry(&self, side: EndpointSide) -> Option<Rect> {
        self.menu_entries[side.index()]
    }

    /// The rows of the open *Switch image…* popup as the last frame drew
    /// them: where each row's image name was drawn, and the image and pixel
    /// the row moves the endpoint to.
    #[cfg(test)]
    pub(crate) fn offer_rows(&self) -> &[(Rect, u32, [f64; 2])] {
        &self.offer_rows
    }

    /// Where endpoint `side`'s pixel field `axis` was drawn on the last frame,
    /// in the edited mode.
    #[cfg(test)]
    pub(crate) fn pixel_field_rect(&self, side: EndpointSide, axis: PixelAxis) -> Option<Rect> {
        self.endpoint_fields[side.index()].pixel_rects[axis.index()]
    }

    /// Where endpoint `side`'s image drop-down was drawn on the last frame, in
    /// the edited mode.
    #[cfg(test)]
    pub(crate) fn image_field_rect(&self, side: EndpointSide) -> Option<Rect> {
        self.endpoint_fields[side.index()].image_rect
    }

    /// The rows of endpoint `side`'s open image drop-down as the last frame
    /// drew them: where each was drawn, and the image it names.
    #[cfg(test)]
    pub(crate) fn image_choices(&self, side: EndpointSide) -> &[(Rect, u32)] {
        &self.endpoint_fields[side.index()].choices
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
        self.blocks = [None, None];
        self.switches = [None, None];
        self.menu_entries = [None, None];
        self.offer_rows.clear();
        for fields in &mut self.endpoint_fields {
            fields.pixel_rects = [None, None];
            fields.image_rect = None;
            fields.choices.clear();
        }
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
        self.drafts = [None, None];
        self.endpoint_fields = Default::default();
        self.offers = None;
        self.offer_crops.clear();
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
                self.show_blocks(ui, state, node, target, &blocks, None, &mut response);
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
                show_item_comparison(ui, row.as_ref(), || {
                    let row = row.as_ref()?;
                    let reading = reading.resolved.as_ref()?;
                    measurements::item_scale_comparison(rows, resolved, origin, row, reading)
                });
                ui.separator();
                let edit = ItemEdit {
                    label,
                    item: &item,
                    refusal: state.bench_edit_refusal(node.id),
                };
                self.show_blocks(
                    ui,
                    state,
                    node,
                    Target::Item(node.id),
                    &blocks,
                    Some(&edit),
                    response,
                );
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
        let (committed, rect) = draft_field(
            ui,
            &mut self.drafts[field.index()],
            DraftField {
                item,
                value,
                refusal,
                empty,
                hint,
                width: 180.0,
                name: field.widget_name(),
            },
        );
        self.fields[field.index()] = Some(rect);
        committed
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

    /// The two endpoint blocks, each with its pictures cut on the first frame
    /// that shows it: side by side where the panel is wide enough for two,
    /// and one above the other where it is not, so neither draws over the
    /// other. In the edited mode (`edit`) each block carries *Switch image…*
    /// and its right-click menu, and the popup the two open.
    #[allow(
        clippy::too_many_arguments,
        reason = "the frame's inputs, read once by show"
    )]
    fn show_blocks(
        &mut self,
        ui: &mut egui::Ui,
        state: &AppState,
        node: &SceneNode,
        target: Target,
        blocks: &[Option<Endpoint>; 2],
        edit: Option<&ItemEdit<'_>>,
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
        let mut show_one = |this: &mut Self, ui: &mut egui::Ui, side: EndpointSide| {
            let switch = edit.map(|edit| {
                switch_refusal(
                    edit.refusal.as_deref(),
                    side,
                    blocks[side.index()].as_ref().map(|e| &e.resolution),
                )
            });
            let Some(endpoint) = &blocks[side.index()] else {
                let drawn = show_unplaced(ui, side, switch);
                this.note_block(side, &drawn);
                return;
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
                switch,
            };
            this.ensure_pictures(ui.ctx(), state, &block, key);
            let pictures = this.pictures.get(&key);
            let fields = edit.map(|edit| BlockFields {
                item: edit.label,
                refusal: edit.refusal.as_deref(),
                kept: &mut this.endpoint_fields[side.index()],
            });
            let drawn = block.show(ui, pictures, fields, response);
            this.names[side.index()] = Some(drawn.name);
            this.note_block(side, &drawn);
            if let (Some(edit), Some(button)) = (edit, &drawn.switch_button) {
                this.show_switch_popup(
                    ui,
                    state,
                    node,
                    edit,
                    endpoint,
                    side,
                    button,
                    drawn.open_switch,
                    response,
                );
            }
        };
        if ui.available_width() >= 2.0 * BLOCK_MIN_WIDTH {
            ui.columns(2, |columns| {
                for side in EndpointSide::BOTH {
                    show_one(self, &mut columns[side.index()], side);
                }
            });
        } else {
            for side in EndpointSide::BOTH {
                show_one(self, ui, side);
            }
        }
    }

    /// Keep where an endpoint block and its *Switch image…* button were
    /// drawn, for the tests that read them.
    fn note_block(&mut self, side: EndpointSide, drawn: &BlockDrawn) {
        self.blocks[side.index()] = Some(drawn.rect);
        self.switches[side.index()] = drawn
            .switch_button
            .as_ref()
            .map(|button| (button.rect, drawn.switch_refusal.clone()));
        self.menu_entries[side.index()] = drawn.menu_entry;
    }

    /// The *Switch image…* popup of endpoint `side`, anchored on its button:
    /// opened by the button or by the block's right-click menu
    /// (`open_from_menu`), and closed by a click outside it or on a row.
    ///
    /// It lists the offers of core's `endpoint_views`, grouped by point, the
    /// chosen candidate's point first. Each row is the image name, a crop of
    /// that image around the observation's square with a cross at the offered
    /// pixel and the endpoint crop's hover view, the observation's `ℓ` and
    /// viewing angle, and a mark where the endpoint would resolve to another
    /// point there, or to none. A click on a row asks for the switch.
    #[allow(
        clippy::too_many_arguments,
        reason = "the block's inputs, each read once"
    )]
    fn show_switch_popup(
        &mut self,
        ui: &egui::Ui,
        state: &AppState,
        node: &SceneNode,
        edit: &ItemEdit<'_>,
        endpoint: &Endpoint,
        side: EndpointSide,
        button: &egui::Response,
        open_from_menu: bool,
        response: &mut MeasurementViewResponse,
    ) {
        let popup = egui::Popup::from_toggle_button_response(button)
            .close_behavior(egui::PopupCloseBehavior::CloseOnClickOutside);
        let popup_id = popup.get_id();
        if open_from_menu {
            egui::Popup::open_id(ui.ctx(), popup_id);
        }
        if !button.enabled() {
            egui::Popup::close_id(ui.ctx(), popup_id);
            return;
        }
        let document = node.history.current_version().document_serial;
        let held = self.offers.as_ref().is_some_and(|held| {
            Arc::ptr_eq(&held.item, edit.item) && held.document == document && held.side == side
        });
        let is_open = egui::Popup::is_id_open(ui.ctx(), popup_id) || button.clicked();
        if is_open && !held {
            self.offer_crops.clear();
            self.offers = Some(HeldOffers {
                item: Arc::clone(edit.item),
                document,
                side,
                views: endpoint_views(
                    node.edited(),
                    endpoint.image,
                    endpoint.pixel,
                    &MeasurementEndpointOptions::default(),
                ),
            });
        }
        let mut clicked = None;
        let shown = popup.show(|ui| {
            let Some(held) = self.offers.as_ref().filter(|held| held.side == side) else {
                return;
            };
            ui.set_max_width(OFFER_CROP_SIDE + 260.0);
            ui.label(
                egui::RichText::new(format!(
                    "Switch endpoint {} to another image",
                    side.letter()
                ))
                .strong(),
            );
            ui.weak(
                "Each row is the same place on the surface, seen in another photograph. \
                 Click one to move the endpoint there.",
            );
            let views = match &held.views {
                Ok(views) => views.clone(),
                Err(refusal) => {
                    ui.colored_label(ui.visuals().warn_fg_color, format!("{refusal}."));
                    return;
                }
            };
            if views.is_empty() {
                ui.weak("No other observation of these points shows this place.");
                return;
            }
            let chosen = endpoint.resolution.as_ref().ok().map(|r| r.chosen().point);
            egui::ScrollArea::vertical()
                .id_salt(("measurement_switch", side.index()))
                .max_height(420.0)
                .show(ui, |ui| {
                    let mut last_point = None;
                    for view in &views {
                        if last_point != Some(view.point) {
                            last_point = Some(view.point);
                            ui.separator();
                            let id = crate::scene::point_id(node, view.point as usize);
                            let which = if Some(view.point) == chosen {
                                "the point the endpoint resolves to"
                            } else {
                                "another square holding the pixel"
                            };
                            ui.label(egui::RichText::new(format!("Point {id}: {which}")).strong());
                        }
                        if self.show_offer(ui, state, node, document, view) {
                            clicked = Some(view.clone());
                        }
                    }
                });
        });
        // A closed popup lets go of its offers and their crops; the other
        // endpoint's popup, if open, keeps its own.
        if shown.is_none() && self.offers.as_ref().is_some_and(|held| held.side == side) {
            self.offers = None;
            self.offer_crops.clear();
        }
        if let Some(view) = clicked {
            egui::Popup::close_id(ui.ctx(), popup_id);
            response.switch_endpoint = Some(EndpointSwitch {
                node: node.id,
                item: edit.label.to_string(),
                side,
                image: view.image,
                pixel: view.pixel,
            });
        }
    }

    /// One row of the *Switch image…* popup. Whether it was clicked.
    fn show_offer(
        &mut self,
        ui: &mut egui::Ui,
        state: &AppState,
        node: &SceneNode,
        document: VersionSerial,
        view: &EndpointView,
    ) -> bool {
        let key = OfferKey {
            node: node.id,
            document,
            point: view.point,
            observation: view.observation,
            pixel: view.pixel.map(f64::to_bits),
        };
        if !self.offer_crops.contains_key(&key) {
            let name = format!(
                "measurement_offer_{:?}_{}_{}_{:.3}_{:.3}",
                node.id, view.point, view.observation, view.pixel[0], view.pixel[1]
            );
            let square = Some((view.point, view.observation));
            if let Some(crop) = endpoint_crop(
                ui.ctx(),
                state,
                node,
                view.image,
                view.pixel,
                square,
                name,
                &self.regions,
            ) {
                self.offer_crops.insert(key, crop);
            }
        }
        let recon = node.recon();
        let mut targets: Vec<egui::Response> = Vec::new();
        ui.horizontal(|ui| {
            targets.push(show_crop(
                ui,
                recon,
                view.image,
                self.offer_crops.get(&key),
                OFFER_CROP_SIDE,
            ));
            ui.vertical(|ui| {
                targets.push(
                    ui.add(
                        egui::Label::new(
                            egui::RichText::new(measurements::image_name(recon, view.image))
                                .strong(),
                        )
                        .sense(egui::Sense::click())
                        .truncate(),
                    ),
                );
                targets.push(
                    ui.add(
                        egui::Label::new(format!("({:.1}, {:.1})", view.pixel[0], view.pixel[1]))
                            .sense(egui::Sense::click()),
                    ),
                );
                let at_infinity = node
                    .edited()
                    .point(view.point)
                    .is_some_and(|p| p.point().w == 0.0);
                targets.push(
                    ui.add(
                        egui::Label::new(offer_readings_text(view, at_infinity, unit_of(node)))
                            .sense(egui::Sense::click()),
                    )
                    .on_hover_text(
                        "\u{2113}: how closely this observation fixes its square on the \
                         surface, its self-similarity length on the patch. \u{03b8}: the angle \
                         it sees the square at, 0\u{00b0} facing it. Smaller is sharper on both.",
                    ),
                );
                if let Some(mark) = offer_mark(node, view) {
                    targets.push(
                        ui.add(
                            egui::Label::new(
                                egui::RichText::new(mark).color(ui.visuals().warn_fg_color),
                            )
                            .sense(egui::Sense::click()),
                        )
                        .on_hover_text(
                            "Another square holds this pixel in that photograph, so the \
                             endpoint would land on a different point there, or on none.",
                        ),
                    );
                }
            });
        });
        self.offer_rows
            .push((targets[1].rect, view.image, view.pixel));
        targets.iter().any(egui::Response::clicked)
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
        let square = block.resolution.as_ref().ok().map(|resolution| {
            let chosen = resolution.chosen();
            (chosen.point, chosen.observation)
        });
        let name = format!(
            "measurement_crop_{:?}_{}_{}",
            block.node.id,
            block.texture_stem(),
            block.side.letter()
        );
        let Some(crop) = endpoint_crop(
            ctx,
            state,
            block.node,
            block.image,
            block.pixel,
            square,
            name,
            &self.regions,
        ) else {
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

/// An endpoint that is not placed: a block saying so, with *Switch image…*
/// greyed in the edited mode (`switch`).
fn show_unplaced(
    ui: &mut egui::Ui,
    side: EndpointSide,
    switch: Option<Option<String>>,
) -> BlockDrawn {
    let mut button = None;
    let mut targets = Vec::new();
    let frame = egui::Frame::group(ui.style()).show(ui, |ui| {
        ui.set_width(ui.available_width());
        targets.push(
            ui.add(
                egui::Label::new(egui::RichText::new(side.letter()).strong())
                    .sense(egui::Sense::click()),
            ),
        );
        targets.push(
            ui.add(
                egui::Label::new(
                    egui::RichText::new(format!("Endpoint {} is not placed.", side.letter()))
                        .color(ui.visuals().weak_text_color()),
                )
                .sense(egui::Sense::click()),
            ),
        );
        if let Some(refusal) = &switch {
            button = Some(switch_button(ui, refusal.clone()));
        }
    });
    let refusal = switch.flatten();
    let mut menu_entry = None;
    if button.is_some() {
        for target in &targets {
            switch_menu(target, refusal.clone(), &mut menu_entry);
        }
    }
    BlockDrawn {
        rect: frame.response.rect,
        name: targets[0].rect,
        switch_button: button,
        switch_refusal: refusal,
        open_switch: false,
        menu_entry,
    }
}

/// What the edited mode adds to the endpoint blocks: the item they show, its
/// label on the bench, and why a step on it would be refused
/// (`AppState::bench_edit_refusal`).
struct ItemEdit<'a> {
    label: &'a str,
    item: &'a Arc<EditableMeasurement>,
    refusal: Option<String>,
}

/// What drawing one endpoint block left: where it and its name line were
/// drawn, its *Switch image…* button in the edited mode with the refusal it
/// was greyed with, and whether its right-click menu asked for the popup.
struct BlockDrawn {
    rect: Rect,
    name: Rect,
    switch_button: Option<egui::Response>,
    switch_refusal: Option<String>,
    open_switch: bool,
    /// Where the right-click menu's *Switch Image…* was drawn, while it is
    /// open.
    menu_entry: Option<Rect>,
}

/// Why *Switch image…* is greyed for endpoint `side`, or `None`: a step on
/// the item refused (`bench_refusal`, a task holding the node or a view-only
/// bench), the endpoint not placed (`resolution` is `None`), or the endpoint
/// refused by the resolver, in which case no square names a place for other
/// images to show.
pub(crate) fn switch_refusal(
    bench_refusal: Option<&str>,
    side: EndpointSide,
    resolution: Option<&Result<EndpointResolution, EndpointRefusal>>,
) -> Option<String> {
    if let Some(why) = bench_refusal {
        return Some(why.to_string());
    }
    let letter = side.letter();
    match resolution {
        None => Some(format!(
            "Endpoint {letter} is not placed, so it has no image to switch from."
        )),
        Some(Err(refusal)) => Some(format!(
            "Endpoint {letter} has no position, so no other image can be offered: {refusal}."
        )),
        Some(Ok(_)) => None,
    }
}

/// An endpoint block's *Switch image…* button, greyed with `refusal` as its
/// hover text where there is one.
fn switch_button(ui: &mut egui::Ui, refusal: Option<String>) -> egui::Response {
    let enabled = refusal.is_none();
    let button = ui.add_enabled(enabled, egui::Button::new(SWITCH_LABEL).small());
    match refusal {
        Some(why) => button.on_disabled_hover_text(why),
        None => button.on_hover_text(
            "Move this endpoint to the same place on the surface in another photograph",
        ),
    }
}

/// The right-click menu of an endpoint block's `target`, which selects
/// nothing (G3): *Switch Image…*, greyed with `refusal`. Whether the entry was
/// clicked; where it was drawn, while the menu is open, goes in `drawn`.
fn switch_menu(target: &egui::Response, refusal: Option<String>, drawn: &mut Option<Rect>) -> bool {
    let mut asked = false;
    crate::context_menu::on_secondary_click(target).show(|ui| {
        let entry = ui.add_enabled(refusal.is_none(), egui::Button::new(SWITCH_ENTRY));
        *drawn = Some(entry.rect);
        let entry = match &refusal {
            Some(why) => entry.on_disabled_hover_text(why),
            None => entry,
        };
        if entry.clicked() {
            asked = true;
            ui.close();
        }
    });
    asked
}

/// The readings line of a popup row: `ℓ` in the file's unit, scene units, or
/// degrees for a point at infinity, and the viewing angle.
pub(crate) fn offer_readings_text(
    view: &EndpointView,
    at_infinity: bool,
    unit: Option<&str>,
) -> String {
    let length = match view.localization {
        None => "\u{2113} not read".to_string(),
        Some(l) if at_infinity => format!("\u{2113} {}\u{00b0}", measurements::significant(l, 3)),
        Some(l) => format!(
            "\u{2113} {} {}",
            measurements::significant(l, 3),
            unit.unwrap_or("su")
        ),
    };
    let angle = view
        .viewing_angle_deg
        .map_or("\u{03b8} unknown".to_string(), |a| {
            format!("\u{03b8} {a:.1}\u{00b0}")
        });
    format!("{length}  {angle}")
}

/// The mark of a popup row whose pixel resolves elsewhere: the point the
/// endpoint would land on there, or that it would have no position.
pub(crate) fn offer_mark(node: &SceneNode, view: &EndpointView) -> Option<String> {
    if !view.resolves_elsewhere() {
        return None;
    }
    Some(match view.resolves_to {
        Some(point) => format!(
            "\u{26a0} resolves to point {} there",
            crate::scene::point_id(node, point as usize)
        ),
        None => "\u{26a0} no position there".to_string(),
    })
}

/// The file's unit as a length unit name, or `None` for scene units.
fn unit_of(node: &SceneNode) -> Option<&str> {
    node.recon()
        .metadata
        .world_space_unit
        .as_deref()
        .filter(|u| LengthUnit::from_name(u).is_some())
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
/// ([`show_item_comparison`]).
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
fn show_item_comparison(
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
    /// In the edited mode, *Switch image…* with the refusal it is greyed
    /// with, if any; `None` for a stored measurement, which has no steps.
    switch: Option<Option<String>>,
}

impl EndpointBlock<'_> {
    /// What the block's textures are named by: the endpoint's image and
    /// pixel, which is all its pictures are cut from.
    fn texture_stem(&self) -> String {
        format!("{}_{:.3}_{:.3}", self.image, self.pixel[0], self.pixel[1])
    }

    /// Draw the block and report its gestures into `response`. In the edited
    /// mode (`fields`) the image name is a drop-down of the reconstruction's
    /// images and the pixel is two number fields
    /// ([`Self::show_fields`]).
    ///
    /// Every line wraps inside the block's width, so a long one, such as the
    /// position, never reaches into the block beside it.
    fn show(
        &self,
        ui: &mut egui::Ui,
        pictures: Option<&EndpointPictures>,
        fields: Option<BlockFields<'_>>,
        response: &mut MeasurementViewResponse,
    ) -> BlockDrawn {
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
        let mut name_rect = Rect::NOTHING;
        let mut button = None;
        let drawn = frame.show(ui, |ui| {
            let width = ui.available_width();
            ui.set_width(width);
            ui.set_max_width(width);
            match fields {
                None => {
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
                    name_rect = name.rect;
                    targets.push(name);
                    targets.push(ui.add(
                        egui::Label::new(format!("({x:.1}, {y:.1})")).sense(egui::Sense::click()),
                    ));
                }
                Some(fields) => {
                    let letter = self.show_fields(ui, fields, response);
                    name_rect = letter.rect;
                    targets.push(letter);
                }
            }
            if let Some(refusal) = &self.switch {
                button = Some(switch_button(ui, refusal.clone()));
            }
            ui.horizontal_wrapped(|ui| {
                targets.push(show_crop(
                    ui,
                    recon,
                    image,
                    pictures.map(|p| &p.crop),
                    CROP_SIDE,
                ));
                if let Some(patch) = pictures.and_then(|p| p.patch.as_ref()) {
                    targets.push(self.show_patch(ui, patch));
                }
            });
            self.show_resolution(ui, response);
        });
        let refusal = self.switch.clone().flatten();
        let mut open_switch = false;
        let mut menu_entry = None;
        if button.is_some() {
            for target in &targets {
                open_switch |= switch_menu(target, refusal.clone(), &mut menu_entry);
            }
        }
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
        BlockDrawn {
            rect: drawn.response.rect,
            name: name_rect,
            switch_button: button,
            switch_refusal: refusal,
            open_switch,
            menu_entry,
        }
    }
}

/// A crop cell `side` points square of image `image`, with its hover view: the
/// photograph's full name, the crop in context with the square's outline and
/// the cross, and the caption. Used by an endpoint block and by each row of
/// the *Switch image…* popup.
fn show_crop(
    ui: &mut egui::Ui,
    recon: &sfmtool_core::SfmrReconstruction,
    image: u32,
    crop: Option<&Crop>,
    side: f32,
) -> egui::Response {
    let (rect, response) = ui.allocate_exact_size(egui::vec2(side, side), egui::Sense::click());
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 2.0, ui.visuals().extreme_bg_color);
    let full_name = measurements::image_name(recon, image);
    match crop {
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
                    let (view, _) =
                        ui.allocate_exact_size(context.fitted(bounds).size(), egui::Sense::hover());
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

/// What the edited mode hands an endpoint block for its fields: the item, by
/// its label on the bench, why a step on it would be refused, which greys
/// them, and what the block keeps between frames.
struct BlockFields<'a> {
    item: &'a str,
    refusal: Option<&'a str>,
    kept: &'a mut EndpointFields,
}

impl EndpointBlock<'_> {
    /// The edited mode's first two lines of a block: the endpoint's letter,
    /// the block's click target on that line, beside a drop-down of the
    /// reconstruction's images, then the pixel as two number fields, `x` and
    /// `y`. The letter's response comes back.
    ///
    /// Picking another image in the drop-down is one step that sets the image
    /// and the pixel the endpoint holds. A pixel field commits as the header's
    /// fields do ([`draft_field`]), as one step that sets the image the
    /// endpoint holds and the pixel with that coordinate typed. Text that is
    /// not a finite number asks for no step and is reported as a refusal.
    /// Both are greyed, with the step's refusal as hover text, while a step
    /// on the item would be refused.
    fn show_fields(
        &self,
        ui: &mut egui::Ui,
        fields: BlockFields<'_>,
        response: &mut MeasurementViewResponse,
    ) -> egui::Response {
        let BlockFields {
            item,
            refusal,
            kept,
        } = fields;
        let recon = self.node.recon();
        let side = self.side;
        let set = |image: u32, pixel: [f64; 2]| crate::bench::measurement::EndpointSet {
            node: self.node.id,
            item: item.to_string(),
            side,
            image,
            pixel,
        };
        let letter = ui
            .horizontal(|ui| {
                let letter = ui.add(
                    egui::Label::new(egui::RichText::new(side.letter()).strong())
                        .sense(egui::Sense::click()),
                );
                let mut chosen = None;
                let width = (ui.available_width() - 8.0).max(80.0);
                let combo = ui
                    .add_enabled_ui(refusal.is_none(), |ui| {
                        egui::ComboBox::from_id_salt(("measurement_endpoint_image", side.index()))
                            .selected_text(measurements::image_name(recon, self.image))
                            .width(width)
                            .truncate()
                            .show_ui(ui, |ui| {
                                for (i, image) in recon.image_set.images.iter().enumerate() {
                                    let i = i as u32;
                                    let row = ui.selectable_label(i == self.image, &image.name);
                                    kept.choices.push((row.rect, i));
                                    if row.clicked() {
                                        chosen = Some(i);
                                    }
                                }
                            })
                    })
                    .inner
                    .response;
                crate::widget_name::set_widget_name(&combo, image_field_name(side));
                kept.image_rect = Some(combo.rect);
                match refusal {
                    Some(why) => {
                        combo.on_disabled_hover_text(why);
                    }
                    None => {
                        combo.on_hover_text(
                            "The photograph this endpoint is in. Picking another keeps the \
                             pixel, which has to be inside that photograph.",
                        );
                    }
                }
                if let Some(image) = chosen.filter(|&image| image != self.image) {
                    response.set_endpoint = Some(set(image, self.pixel));
                }
                letter
            })
            .inner;
        ui.horizontal(|ui| {
            for axis in PixelAxis::BOTH {
                ui.label(axis.letter());
                let value = pixel_text(self.pixel[axis.index()]);
                let hint = format!(
                    "The endpoint's {} pixel coordinate. Enter or clicking elsewhere sets it; \
                     Escape puts it back.",
                    axis.letter()
                );
                let (committed, rect) = draft_field(
                    ui,
                    &mut kept.drafts[axis.index()],
                    DraftField {
                        item,
                        value: &value,
                        refusal,
                        empty: "",
                        hint: &hint,
                        width: 72.0,
                        name: pixel_field_name(side, axis),
                    },
                );
                kept.pixel_rects[axis.index()] = Some(rect);
                let Some(text) = committed else {
                    continue;
                };
                match text.trim().parse::<f64>().ok().filter(|v| v.is_finite()) {
                    Some(coordinate) => {
                        let mut pixel = self.pixel;
                        pixel[axis.index()] = coordinate;
                        response.set_endpoint = Some(set(self.image, pixel));
                    }
                    None => {
                        response.endpoint_refused = Some(format!(
                            "Cannot move endpoint {} of {item}: \"{text}\" is not a number.",
                            side.letter()
                        ));
                    }
                }
            }
        });
        letter
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

/// The crop around pixel `pixel` of image `image`, or `None` while its
/// photograph is decoding: an endpoint's, or a *Switch image…* row's.
///
/// Where `square` names an observation (a point and its index in the
/// point's track), the crop is cut around that observation's square in the
/// image, as Track View's crop is cut around a row's square, with a cross at
/// the pixel. Where it is `None`, the endpoint does not resolve, and the crop
/// is a [`UNRESOLVED_CROP_PX`] square of the photograph centred on the pixel,
/// with no outline. `name` names its textures.
#[allow(
    clippy::too_many_arguments,
    reason = "what the crop is cut from, each read once"
)]
fn endpoint_crop(
    ctx: &egui::Context,
    state: &AppState,
    node: &SceneNode,
    image: u32,
    pixel: [f64; 2],
    square: Option<(u32, usize)>,
    name: String,
    regions: &PanelRegionPool,
) -> Option<Crop> {
    let recon = node.recon();
    let image = image as usize;
    if image >= recon.image_set.images.len() {
        return Some(Crop::Absent("The image is not in the reconstruction"));
    }
    let square = square.and_then(|(point, observation)| {
        let view = node.edited().point(point)?;
        let patch_frame = view.patch_frame()?;
        let keypoint = view.keypoint_xy(observation)?;
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
