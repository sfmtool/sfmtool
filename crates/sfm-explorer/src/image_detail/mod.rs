// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Image detail panel — full-resolution image display for the selected camera,
//! with SIFT feature overlays and heatmap visualization modes.
//!
//! [`ImageDetail::show`] orchestrates the panel each frame; the heavier pieces
//! live in sibling modules:
//! - [`input`] — drag/scroll/pinch/keyboard/gesture view manipulation.
//! - [`overlay`] — the feature-overlay draw modes, hit-testing, and tooltip.
//! - [`mod@intrinsics`] — the intrinsics overlay layer, drawn independently of
//!   the feature mode and composing with whichever one is active.
//! - [`mod@bench_track`]: the bench layer, the active editable track drawn
//!   over everything else in the bench's own colours.
//! - [`mod@view`] -- what the pan and the zoom mean, as a pure function over the
//!   frame's geometry, so the row-click reveal and the wire's
//!   `set_image_detail_view` are one computation.

mod bench_track;
mod input;
mod intrinsics;
mod overlay;
#[cfg(test)]
mod tests;
mod view;

pub(crate) use intrinsics::{show_intrinsics_controls, CameraLayer};
pub(crate) use overlay::{BenchMenu, START_CLUSTER_LABEL};
pub(crate) use view::{look_at, Look, ViewGeometry};

use crate::document::VersionSerial;
use crate::platform::{GestureEvent, ScrollInput};
use crate::scene::{CameraRef, ImageRef, PointRef, ReconId};
use crate::state::edits::PointGesture;
use crate::state::{
    CachedSiftFeatures, FeatureDisplaySettings, IntrinsicsDisplaySettings, OverlayMode,
};
use crate::texture::rgb_to_color_image;
use sfmtool_core::camera::remap::ImageU8;
use sfmtool_core::camera::CameraIntrinsics;
use sfmtool_core::spatial::PointCloud2;
use sfmtool_core::EditedReconstruction;
use std::collections::HashMap;

use intrinsics::View;

/// Maximum zoom level (32× = pixel-level inspection).
pub(crate) const MAX_ZOOM: f32 = 32.0;
/// Minimum overlap in pixels between image and panel when panning.
const PAN_MARGIN: f32 = 50.0;
/// How far in from the panel's edge a revealed pixel still counts as out of
/// view, as a fraction of the panel's size per axis. See
/// [`view::Look::Reveal`].
const REVEAL_MARGIN: f32 = 0.05;

/// Prepared feature overlay state for the current image in the detail panel.
struct FeatureOverlayState {
    /// The image this overlay was built for. A ref, so a file replacement that
    /// leaves the same index selected still invalidates it.
    image: ImageRef,
    /// The node version this overlay was built for. A point edit leaves the
    /// image table alone, so nothing here is keyed by an index that moved, but
    /// it does change which points are live, and the overlay is a statement
    /// about the live ones: the features it draws, and the point each one
    /// selects. Serials are minted once and never reused, so a new version and
    /// a cursor move back onto an old one both arrive here as a different
    /// value.
    version: VersionSerial,
    overlay_mode: OverlayMode,
    tracked_only: bool,
    max_features: Option<usize>,
    min_feature_size: Option<f32>,
    max_feature_size: Option<f32>,
    features: Vec<DisplayFeature>,
    tree: PointCloud2<f32>,
}

/// Image detail panel state.
pub struct ImageDetail {
    /// Currently loaded full-res image and the texture built from it.
    loaded_image: Option<(ImageRef, egui::TextureHandle)>,
    /// Prepared feature overlay for the current image.
    feature_overlay: Option<FeatureOverlayState>,
    /// The intrinsics layer's per-camera products, keyed by [`CameraRef`].
    ///
    /// Bounded by the number of *distinct* intrinsics across the loaded nodes,
    /// which is small even for a per-image-intrinsics solve, so there is no
    /// eviction beyond [`ImageDetail::forget_recon`].
    intrinsics: HashMap<CameraRef, CameraLayer>,
    /// The pixel the context menu was last opened at, in source-image
    /// coordinates, for the entries that act at it.
    ///
    /// The menu's entries are laid out on later frames, by which time the
    /// pointer has moved off the place the user named, so what an entry reads
    /// is what the opening frame recorded. The bench entries carry it back out
    /// in their own response fields.
    menu_pixel: Option<[f32; 2]>,
    /// Offset of image center from panel center, in panel pixels.
    pan: egui::Vec2,
    /// Zoom level. 1.0 = fit image to panel. >1.0 = zoomed in.
    zoom: f32,
    /// Displayed image extent that the current [`ImageDetail::pan`] was
    /// measured against, from the last frame that drew one. `None` until a
    /// frame has drawn, and again after a view reset. See
    /// [`ImageDetail::rescale_view`].
    last_display_size: Option<egui::Vec2>,
    /// The bench layer's handle the pointer has hold of, while it has one.
    ///
    /// Held on the panel rather than derived each frame because a drag is a
    /// gesture and not a state of the pointer: what is being dragged was
    /// decided when the button went down, and the pointer wanders off the
    /// handle the moment it starts moving.
    bench_drag: Option<bench_track::Drag>,
    /// What a gesture on a feature asked of the app, drained by `app.rs` after
    /// the frame.
    ///
    /// Held here rather than carried out in the panel's response because it
    /// ends in a layout operation: Edit on Bench raises Track View, and the
    /// frame swaps the dock out of the state while a tab body draws. See
    /// [`ImageDetail::take_point_gesture`].
    point_gesture: Option<PointGesture>,
    /// *Start cluster on the bench here*, at the image and pixel the menu was
    /// opened on, drained by `app.rs` after the frame for the same reason: it
    /// raises Track View on the cluster. See [`ImageDetail::take_cluster_start`].
    cluster_start: Option<(ImageRef, [f32; 2])>,
}

/// A feature to draw on the image detail panel.
struct DisplayFeature {
    /// Feature position in image pixel coordinates (x, y).
    position: [f32; 2],
    /// 2x2 affine shape matrix [[a11, a12], [a21, a22]].
    affine_shape: [[f32; 2]; 2],
    /// The 3D point index this feature maps to, or `u32::MAX` if untracked.
    point_index: u32,
    /// Max pairwise angle (degrees) between observing rays for this feature's
    /// 3D point — the track's widest triangulation baseline. NaN for untracked
    /// features or when not populated.
    max_track_angle_deg: f32,
    /// Inverse-depth z-score (`depth / σ_depth`) of this feature's 3D point.
    /// NaN for untracked / infinity points or when not populated.
    inverse_depth_z: f32,
    /// Condition number of this feature's 3D point's triangulation normal
    /// matrix. NaN for untracked / infinity points or when not populated.
    condition_number: f32,
}

impl DisplayFeature {
    fn is_tracked(&self) -> bool {
        self.point_index != u32::MAX
    }
}

/// Sentinel value for untracked features.
const UNTRACKED: u32 = u32::MAX;

/// Response from the image detail panel.
///
/// Point indices are local to the reconstruction the panel was shown with;
/// `dock.rs` pairs them back into [`crate::scene::PointRef`]s.
pub struct ImageDetailResponse {
    /// If Some, the user clicked a feature — select this 3D point.
    pub select_point: Option<usize>,
    /// Point index currently under the pointer (for cross-panel hover).
    pub hovered_point: Option<usize>,
    /// Whether the pointer is currently inside the detail panel.
    pub has_pointer: bool,
    /// The pixel a right-click just opened the context menu at, in
    /// source-image coordinates. Set on the frame the menu opens and kept by
    /// the panel, because the menu's entries are drawn a frame later, by which
    /// time the pointer has moved.
    pub context_menu_pixel: Option<[f32; 2]>,
    /// The pixel the context menu's `Start cluster on the bench here` was
    /// clicked for: a cluster-stage track starts there, on the node's bench.
    ///
    /// Read by the panel itself, like `edit_on_bench`: the gesture raises
    /// Track View, so `show` keeps it for [`ImageDetail::take_cluster_start`].
    pub start_bench_cluster: Option<[f32; 2]>,
    /// The pixel the context menu's `Add observation to bench track here` was
    /// clicked for: a candidate joins the bench's active track there.
    pub add_bench_observation: Option<[f32; 2]>,
    /// `Edit on Bench` was chosen on a feature, or one was double-clicked:
    /// put the point it observes on the bench and raise Track View.
    ///
    /// Read by the panel itself rather than by the dock, because the gesture
    /// ends in a layout operation and so has to outlive the tab body: `show`
    /// turns it into the [`PointGesture`] `app.rs` drains after the frame.
    pub edit_on_bench: Option<usize>,
    /// A mark of the bench layer was clicked: select this observation's row of
    /// the active track in Track View. The layer is on top, so a click it
    /// catches leaves `select_point` alone.
    pub select_bench_row: Option<usize>,
    /// The edit a drag of one of the bench layer's handles just finished, in
    /// the form the core steps take. The dock applies it through
    /// `AppState::edit_bench_patch`, which is the call the wire's eight patch
    /// tools make: one version, one Action Log row, one undo.
    pub bench_edit: Option<crate::bench::PatchEdit>,
    /// What the panel ended this frame looking at, or `None` on a frame that
    /// drew no image.
    ///
    /// Published rather than asked for: the panel is the only thing that knows
    /// how big its body is, and it knows that only while it is drawing. The
    /// dock puts it on `AppState`, where the wire's view tools read it
    /// ([`mod@view`]).
    pub view: Option<ViewGeometry>,
}

impl ImageDetail {
    pub fn new() -> Self {
        Self {
            loaded_image: None,
            feature_overlay: None,
            intrinsics: HashMap::new(),
            menu_pixel: None,
            pan: egui::Vec2::ZERO,
            zoom: 1.0,
            last_display_size: None,
            bench_drag: None,
            point_gesture: None,
            cluster_start: None,
        }
    }

    /// Take what this panel's last frame asked of the app, if anything.
    ///
    /// Drained by `app.rs` once the dock is back in the state, for the reason
    /// the viewport's point menu is drained there: applied inside the tab body
    /// the raise would land on the placeholder dock and be thrown away with it.
    pub(crate) fn take_point_gesture(&mut self) -> Option<PointGesture> {
        self.point_gesture.take()
    }

    /// Take the *Start cluster on the bench here* the last frame chose, if
    /// any: the image and the pixel in it. Drained by `app.rs` with
    /// [`ImageDetail::take_point_gesture`], and for the same reason.
    pub(crate) fn take_cluster_start(&mut self) -> Option<(ImageRef, [f32; 2])> {
        self.cluster_start.take()
    }

    /// Drop everything cached for a reconstruction that has left the scene.
    ///
    /// A [`ReconId`] is never reused, so these entries could only ever go stale
    /// rather than alias — this is about the GPU texture they hold, not about
    /// correctness.
    pub fn forget_recon(&mut self, id: ReconId) {
        if self
            .loaded_image
            .as_ref()
            .is_some_and(|(image, _)| image.recon == id)
        {
            self.loaded_image = None;
        }
        if self
            .feature_overlay
            .as_ref()
            .is_some_and(|overlay| overlay.image.recon == id)
        {
            self.feature_overlay = None;
        }
        self.intrinsics.retain(|camera, _| camera.recon != id);
    }

    /// The cached intrinsics-layer report for `camera`, rebuilt when the camera
    /// or the grid density changes.
    ///
    /// Called from the toolbar as well as from the draw pass, so the popup's
    /// footer and the on-image legend are reading one computation rather than
    /// two that could disagree.
    pub(crate) fn intrinsics_layer(
        &mut self,
        camera_ref: CameraRef,
        camera: &CameraIntrinsics,
        grid_cols: usize,
    ) -> &mut CameraLayer {
        let stale = self
            .intrinsics
            .get(&camera_ref)
            .is_none_or(|layer| layer.grid.0 != grid_cols.max(1));
        if stale {
            self.intrinsics
                .insert(camera_ref, CameraLayer::compute(camera, grid_cols));
        }
        self.intrinsics
            .get_mut(&camera_ref)
            .expect("just inserted when stale")
    }

    /// Reset pan and zoom to fit the image in the panel.
    fn reset_view(&mut self) {
        self.pan = egui::Vec2::ZERO;
        self.zoom = 1.0;
        // Nothing left to carry, so skip the next frame's rescale rather than
        // measure a zero pan against a stale extent.
        self.last_display_size = None;
    }

    /// Hold the framed region of the image fixed when the displayed extent
    /// changes.
    ///
    /// The view deliberately outlives the image it was set on: switching
    /// images, switching reconstructions and resizing the panel all keep
    /// whatever region was being inspected, so two images can be compared by
    /// flipping between them while zoomed in. `pan` alone cannot do that — it
    /// is in panel pixels, so the same value frames a different part of an
    /// image of another resolution. What does survive is `pan / display_size`:
    /// the offset of the image centre from the panel centre as a fraction of
    /// the displayed image, which is exactly the normalized image coordinate
    /// `0.5 - pan / display_size` sitting at the panel centre. Rescaling `pan`
    /// by the extent ratio holds that coordinate fixed, per axis so a change of
    /// aspect ratio is handled too. In the common case — two images of equal
    /// size in an unchanged panel — the ratio is 1 and the view carries over
    /// untouched.
    fn rescale_view(&mut self, display_size: egui::Vec2) {
        let Some(previous) = self.last_display_size else {
            return;
        };
        if previous.x <= 0.0 || previous.y <= 0.0 {
            return;
        }
        self.pan.x *= display_size.x / previous.x;
        self.pan.y *= display_size.y / previous.y;
    }

    /// Apply zoom centered at a cursor position (in panel coordinates relative to panel center).
    fn zoom_at(&mut self, zoom_factor: f32, cursor_rel: egui::Vec2) {
        let old_zoom = self.zoom;
        self.zoom = (self.zoom * zoom_factor).clamp(1.0, MAX_ZOOM);
        let ratio = self.zoom / old_zoom;
        // Adjust pan so the point under the cursor stays fixed.
        self.pan = self.pan * ratio + cursor_rel * (1.0 - ratio);
    }

    /// Point the view at what `look` names, in the frame `geometry` describes.
    ///
    /// The one door a request from outside the panel comes through: the
    /// row-click reveal from both modes of Track View, and the wire's
    /// `set_image_detail_view`. What each request *means* is
    /// [`view::look_at`]'s, a pure function over the geometry, so a pixel an
    /// agent asked to be centred lands exactly where a reveal of the same pixel
    /// would put it; what is here is only the writing of the answer back onto
    /// the fields.
    fn look(&mut self, look: &Look, geometry: ViewGeometry) {
        let next = view::look_at(geometry, look);
        self.pan = egui::vec2(next.pan[0], next.pan[1]);
        self.zoom = next.zoom;
    }

    /// Run the bench layer's handles for this frame: pick a drag up at the
    /// press, follow it, cancel it or finish it, and say which handle the
    /// pointer is on.
    ///
    /// Called before the view's own input so that the pan can be suppressed
    /// from the press onward. The hit test is against the geometry this frame
    /// starts from, which is the geometry the person pressed on: the view has
    /// not moved yet, and it will not move while the button is down over a
    /// handle.
    ///
    /// Both ends of a drag are recorded in **source-image** pixels, so panning
    /// or zooming mid-drag moves the handle with the photograph rather than
    /// under the pointer.
    #[allow(clippy::too_many_arguments)]
    fn update_bench_drag(
        &mut self,
        ui: &egui::Ui,
        interact_response: &egui::Response,
        track: Option<&sfmtool_core::bench::EditableTrack>,
        lock: bool,
        image_table: &sfmtool_core::ImageTable,
        img_idx: usize,
        image_rect: egui::Rect,
        effective_scale: f32,
        response: &mut ImageDetailResponse,
    ) -> Option<bench_track::Handle> {
        // A drag that outlives the image it started in, or the track it was
        // editing, is dropped: its handle names something that is not up.
        if track.is_none() || self.bench_drag.is_some_and(|drag| drag.image != img_idx) {
            self.bench_drag = None;
        }
        let track = track?;
        let layer = bench_track::Layer::build(
            image_table,
            img_idx,
            track,
            image_rect,
            effective_scale,
            lock,
        )?;
        let to_image = |pos: egui::Pos2| -> [f64; 2] {
            [
                f64::from((pos.x - image_rect.min.x) / effective_scale),
                f64::from((pos.y - image_rect.min.y) / effective_scale),
            ]
        };
        let (pressed, down, pointer) = ui.input(|i| {
            (
                i.pointer.primary_pressed(),
                i.pointer.primary_down(),
                i.pointer.interact_pos().or(i.pointer.hover_pos()),
            )
        });

        // **The press decides the handle, not the drag.** egui calls a gesture
        // a drag only once the pointer has left the press by a few pixels, but
        // the view pans on whatever motion it is given, with no threshold of
        // its own. Waiting for `drag_started` therefore lost the gesture twice
        // over: the photograph had already moved, so the handle was no longer
        // under the press position the hit test was given, and the pan had
        // already begun. So the handle is taken at the press, against the
        // geometry this frame starts from -- which is the geometry the person
        // pressed on, since nothing has panned yet -- and from that moment the
        // pan is suppressed.
        if self.bench_drag.is_none() && pressed && !crate::platform::other_mouse_button_down() {
            if let Some((press, handle)) = pointer
                .filter(|_| interact_response.contains_pointer())
                .and_then(|press| layer.hit(press).map(|handle| (press, handle)))
            {
                self.bench_drag = Some(bench_track::Drag {
                    image: img_idx,
                    handle,
                    from: to_image(press),
                    to: to_image(press),
                    moved: false,
                    cancelled: false,
                });
            }
        }
        if let Some(drag) = &mut self.bench_drag {
            if let Some(pos) = interact_response.interact_pointer_pos().or(pointer) {
                let to = to_image(pos);
                drag.moved |= to != drag.from;
                drag.to = to;
            }
            // Escape abandons the gesture. The drag is kept until the button
            // comes up so the view does not start panning halfway through it.
            if ui.input(|i| i.key_pressed(egui::Key::Escape)) {
                drag.cancelled = true;
            }
        }
        // The button coming up ends it, whether or not egui ever called it a
        // drag: a press that never moved is a **click**, which selects the row
        // under it and leaves the track alone.
        if !down {
            if let Some(drag) = self.bench_drag.take() {
                if drag.moved {
                    response.bench_edit = bench_track::Layer::edit(image_table, track, &drag, lock);
                }
            }
        }
        self.bench_drag.map(|drag| drag.handle).or_else(|| {
            ui.input(|i| i.pointer.hover_pos())
                .filter(|_| interact_response.contains_pointer())
                .and_then(|pos| layer.hit(pos))
        })
    }

    /// The view and the frame it is held in, as [`mod@view`] states them.
    fn geometry(
        &self,
        image: ImageRef,
        image_size: egui::Vec2,
        panel_size: egui::Vec2,
    ) -> ViewGeometry {
        ViewGeometry {
            image,
            image_size: [image_size.x, image_size.y],
            panel_size: [panel_size.x, panel_size.y],
            pan: [self.pan.x, self.pan.y],
            zoom: self.zoom,
        }
    }

    /// Clamp pan so the image overlaps the panel by at least PAN_MARGIN pixels.
    fn clamp_pan(&mut self, display_size: egui::Vec2, panel_size: egui::Vec2) {
        let max_pan_x = (display_size.x + panel_size.x) / 2.0 - PAN_MARGIN;
        let max_pan_y = (display_size.y + panel_size.y) / 2.0 - PAN_MARGIN;
        self.pan.x = self.pan.x.clamp(-max_pan_x, max_pan_x);
        self.pan.y = self.pan.y.clamp(-max_pan_y, max_pan_y);
    }

    /// Show the image detail panel.
    #[allow(clippy::too_many_arguments)]
    pub fn show(
        &mut self,
        ui: &mut egui::Ui,
        edited: &EditedReconstruction,
        recon_id: ReconId,
        version: VersionSerial,
        selected_image: Option<usize>,
        // Where a caller has asked the panel to look on this frame, in
        // `selected_image`'s own source pixels. See [`ImageDetail::look`].
        look: Option<Look>,
        selected_point: Option<usize>,
        hovered_point: Option<usize>,
        bench: BenchMenu<'_>,
        gesture_events: &[GestureEvent],
        scroll_input: &ScrollInput,
        sift_features: Option<&CachedSiftFeatures>,
        full_res: Option<&ImageU8>,
        feature_display: &FeatureDisplaySettings,
        intrinsics_display: &mut IntrinsicsDisplaySettings,
    ) -> ImageDetailResponse {
        let mut response = ImageDetailResponse {
            select_point: None,
            hovered_point: None,
            has_pointer: false,
            context_menu_pixel: None,
            start_bench_cluster: None,
            add_bench_observation: None,
            edit_on_bench: None,
            select_bench_row: None,
            bench_edit: None,
            view: None,
        };

        // If no image selected, show placeholder
        let Some(img_idx) = selected_image else {
            ui.centered_and_justified(|ui| {
                ui.label("No image selected");
            });
            if self.loaded_image.is_some() {
                self.loaded_image = None;
                self.feature_overlay = None;
            }
            return response;
        };

        let image_ref = ImageRef::new(recon_id, img_idx);

        // Load the full-resolution image if it changed. The CPU pixels come
        // from the shared `full_res_cache` (decoded once, in dock.rs); this
        // panel only uploads them to a GPU texture.
        if self.loaded_image.as_ref().map(|(i, _)| *i) != Some(image_ref) {
            self.load_image(ui.ctx(), full_res, image_ref);
            self.feature_overlay = None; // reset overlay on image change
        }

        // Determine whether to show features based on overlay mode
        let show_features = feature_display.overlay_mode != OverlayMode::None;

        // Rebuild the overlay when the settings it was built under changed
        // (mode, filters) or when the value underneath it did. The version is
        // in the key because an edit that renumbers nothing still moves what
        // the overlay says: a deleted point's features have to stop being drawn
        // and stop being selectable, or a click lands on an index the version
        // no longer has a point at.
        let cache_valid = self.feature_overlay.as_ref().is_some_and(|c| {
            c.image == image_ref
                && c.version == version
                && c.overlay_mode == feature_display.overlay_mode
                && c.tracked_only == feature_display.tracked_only
                && c.max_features == feature_display.max_features
                && c.min_feature_size == feature_display.min_feature_size
                && c.max_feature_size == feature_display.max_feature_size
        });
        if show_features && !cache_valid {
            self.load_display_features(edited, image_ref, version, sift_features, feature_display);
        } else if !show_features {
            // In None mode, still load tracked features for selected point display
            let tracked_overlay_valid = self
                .feature_overlay
                .as_ref()
                .is_some_and(|c| c.image == image_ref && c.version == version && c.tracked_only);
            if !tracked_overlay_valid {
                self.load_tracked_features(edited, image_ref, version, sift_features);
            }
        }

        // Display the image fitted to the panel. Read the texture's size and id
        // out here rather than holding the handle: the view bookkeeping below
        // needs `&mut self`.
        let Some((tex_size, texture_id)) = self
            .loaded_image
            .as_ref()
            .map(|(_, texture)| (texture.size_vec2(), texture.id()))
        else {
            ui.centered_and_justified(|ui| {
                ui.label("Failed to load image");
            });
            return response;
        };

        let panel_rect = ui.available_rect_before_wrap();
        let panel_size = panel_rect.size();
        let panel_center = panel_rect.center();

        // Base scale: fits the image to the panel at zoom=1.0
        let base_scale = (panel_size.x / tex_size.x).min(panel_size.y / tex_size.y);
        let effective_scale = base_scale * self.zoom;
        let display_size = egui::vec2(tex_size.x * effective_scale, tex_size.y * effective_scale);

        // Carry the framed region across an image or reconstruction switch and
        // across a panel resize — all of which reach here as a change of extent.
        self.rescale_view(display_size);
        self.clamp_pan(display_size, panel_size);

        // A row click elsewhere named a feature in this image, or a tool asked
        // for a place, a rectangle or the whole frame; look there. After the
        // rescale and the clamp, because both are statements about the view
        // this frame starts from and a reveal's test is whether *that* view
        // holds the pixel.
        if let Some(look) = look {
            self.look(&look, self.geometry(image_ref, tex_size, panel_size));
        }

        // Re-derived, because a look that named a zoom moved it.
        let effective_scale = base_scale * self.zoom;
        let display_size = egui::vec2(tex_size.x * effective_scale, tex_size.y * effective_scale);

        // Image rect with pan offset
        let image_center = panel_center + self.pan;
        let image_rect = egui::Rect::from_center_size(image_center, display_size);

        // Allocate the full panel rect for interaction (not just the image rect),
        // so we can handle scroll/drag even when the image is smaller than the panel.
        let interact_rect = panel_rect;
        let interact_id = ui.id().with("image_detail_interact");
        let interact_response =
            ui.interact(interact_rect, interact_id, egui::Sense::click_and_drag());
        response.has_pointer = interact_response.hovered();

        // Draw the image (clipped to panel)
        let painter = ui.painter_at(panel_rect);
        painter.image(
            texture_id,
            image_rect,
            egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
            egui::Color32::WHITE,
        );

        // --- The bench layer's handles, before the view's own input ---
        //
        // A drag that began on a handle is an edit of the track and must not
        // also pan the photograph: the pointer can only mean one of the two,
        // and what it means was decided where the button went down.
        let hovered_handle = self.update_bench_drag(
            ui,
            &interact_response,
            bench.active_track,
            bench.lock,
            &edited.base.image_table,
            img_idx,
            image_rect,
            effective_scale,
            &mut response,
        );

        // What a double-click means, settled before the view input runs and
        // against the geometry the gesture was made on: on a feature that
        // observes a point it is Edit on Bench, and anywhere else it is the
        // zoom `handle_input` applies.
        let double_click_point = interact_response
            .double_clicked()
            .then(|| self.point_under_pointer(ui, image_rect, effective_scale))
            .flatten();
        response.edit_on_bench = double_click_point;

        // --- Input handling ---
        self.handle_input(
            ui,
            &interact_response,
            panel_rect,
            panel_center,
            panel_size,
            display_size,
            scroll_input,
            gesture_events,
            self.bench_drag.is_some(),
            double_click_point.is_some(),
        );

        // Recompute image rect after pan/zoom changes from input
        let effective_scale = base_scale * self.zoom;
        let display_size = egui::vec2(tex_size.x * effective_scale, tex_size.y * effective_scale);
        // The extent `pan` is now measured against, for the next frame's rescale.
        self.last_display_size = Some(display_size);
        // And the view this frame settled on, for the wire, which has no other
        // way to know how big this panel's body is.
        response.view = Some(self.geometry(image_ref, tex_size, panel_size));
        let image_center = panel_center + self.pan;
        let image_rect = egui::Rect::from_center_size(image_center, display_size);

        // --- Intrinsics layer, beneath the features ---
        //
        // `I` toggles it while the pointer is over the panel, alongside the
        // panel's existing `Z`: it is a control users flip constantly once it
        // composes, which is the whole reason it is a layer.
        if response.has_pointer && ui.input(|i| i.key_pressed(egui::Key::I)) {
            intrinsics_display.enabled = !intrinsics_display.enabled;
        }
        // The image table and its cameras are the base's: no point edit moves an
        // image, and a bulk edit that does hands the node a new base.
        let image_table = &edited.base.image_table;
        let camera_ref = image_table
            .images
            .get(img_idx)
            .map(|image| CameraRef::new(recon_id, image.camera_index as usize));
        let camera = camera_ref
            .and_then(|camera_ref| image_table.cameras.get(camera_ref.index()))
            .cloned();
        let view = View {
            origin: image_rect.min,
            scale: base_scale * self.zoom,
            fit: base_scale,
        };
        let intrinsics_readout = match (intrinsics_display.enabled, camera_ref, &camera) {
            (true, Some(camera_ref), Some(camera)) => self.draw_intrinsics(
                &painter,
                camera_ref,
                camera,
                &view,
                panel_rect,
                intrinsics_display,
                ui.input(|i| i.pointer.hover_pos()),
            ),
            _ => None,
        };

        // --- Feature overlays ---
        self.draw_overlays(
            ui,
            &painter,
            &interact_response,
            edited,
            feature_display,
            selected_point,
            hovered_point,
            bench,
            image_rect,
            panel_rect,
            effective_scale,
            intrinsics_readout.as_deref(),
            &mut response,
        );

        // What the entries drawn on later frames act at. Recorded after the
        // draw, so an entry clicked on the frame the menu opened at reads the
        // same pixel the next frame's would.
        if let Some(pixel) = response.context_menu_pixel {
            self.menu_pixel = Some(pixel);
        }

        // The one gesture this panel cannot answer inside a tab body: the
        // point index the menu entry or the double-click named becomes the
        // request `app.rs` applies once the dock is back in the state.
        if let Some(point) = response.edit_on_bench {
            self.point_gesture = Some(PointGesture::EditOnBench(PointRef::new(recon_id, point)));
        }
        // The other one: a cluster started here raises Track View on it.
        if let (Some(pixel), Some(image)) = (response.start_bench_cluster, selected_image) {
            self.cluster_start = Some((ImageRef::new(recon_id, image), pixel));
        }

        // The one mark that draws over the features rather than under them.
        if let Some(camera) = &camera {
            if intrinsics_display.enabled {
                intrinsics::draw_principal_point(&painter, camera, &view, panel_rect);
            }
        }

        // The bench layer, last and over everything: it is about the track
        // being worked on rather than about the reconstruction, so no overlay
        // mode turns it off and none of them draws on top of it.
        if let Some(track) = bench.active_track {
            bench_track::draw(
                &painter,
                ui,
                &interact_response,
                edited,
                img_idx,
                track,
                image_rect,
                effective_scale,
                self.bench_drag.as_ref(),
                hovered_handle,
                bench.lock,
                &mut response,
            );
        }

        response
    }

    /// Build the display texture from the shared full-res CPU image (decoded
    /// once into `AppState::full_res_cache`). `None` means the decode failed,
    /// in which case the "Failed to load image" placeholder path applies.
    fn load_image(&mut self, ctx: &egui::Context, full_res: Option<&ImageU8>, image: ImageRef) {
        let Some(img) = full_res else {
            self.loaded_image = None;
            return;
        };
        let img_idx = image.index();
        // Expand 3-channel RGB to RGBA for the GPU upload.
        let (w, h) = (img.width() as usize, img.height() as usize);
        let color_image = rgb_to_color_image(img.data(), [w, h]);
        let texture = ctx.load_texture(
            format!("detail_{img_idx}"),
            color_image,
            egui::TextureOptions::LINEAR,
        );
        self.loaded_image = Some((image, texture));
    }

    /// Build tracked-only feature list from the shared SIFT cache (for None overlay mode).
    fn load_tracked_features(
        &mut self,
        edited: &EditedReconstruction,
        image: ImageRef,
        version: VersionSerial,
        cached_sift: Option<&CachedSiftFeatures>,
    ) {
        let recon = &*edited.base;
        let img_idx = image.index();
        // Embedded-patches reconstructions keep keypoints inline (no `.sift`
        // cache, empty `image_feature_to_point`); build the tracked-feature list
        // from the per-observation keypoints. Every embedded observation belongs
        // to a point, so all are tracked.
        if recon.feature_indexes().is_none() {
            let features = embedded_image_features(edited, img_idx);
            let tree = build_feature_tree(&features);
            log::info!(
                "Loaded {} embedded tracked features for image {}",
                features.len(),
                img_idx,
            );
            self.feature_overlay = Some(FeatureOverlayState {
                image,
                version,
                overlay_mode: OverlayMode::None,
                tracked_only: true,
                max_features: None,
                min_feature_size: None,
                max_feature_size: None,
                features,
                tree,
            });
            return;
        }

        let feature_to_point = &recon.point_set.image_feature_to_point[img_idx];
        if feature_to_point.is_empty() || cached_sift.is_none() {
            self.feature_overlay = Some(FeatureOverlayState {
                image,
                version,
                overlay_mode: OverlayMode::None,
                tracked_only: true,
                max_features: None,
                min_feature_size: None,
                max_feature_size: None,
                features: Vec::new(),
                tree: PointCloud2::<f32>::new(&[], 0),
            });
            return;
        }
        let cached = cached_sift.unwrap();
        let num_features = cached.positions_xy.len();
        let mut features = Vec::with_capacity(feature_to_point.len());
        for (&feat_idx, &point_idx) in feature_to_point {
            let fi = feat_idx as usize;
            // The map is the base's, so a row it still names may be a point
            // this version deleted (no feature at all, it is the deleted point
            // that must stop being drawn) or one an edit replaced (the same
            // feature, selecting the addition that superseded it).
            let Some(live) = edited.live_index_of_base(point_idx) else {
                continue;
            };
            if fi < num_features {
                features.push(DisplayFeature {
                    position: cached.positions_xy[fi],
                    affine_shape: cached.affine_shapes[fi],
                    point_index: live,
                    max_track_angle_deg: f32::NAN,
                    inverse_depth_z: f32::NAN,
                    condition_number: f32::NAN,
                });
            }
        }
        let tree = build_feature_tree(&features);
        log::info!(
            "Loaded {} tracked features for image {}",
            features.len(),
            img_idx,
        );
        self.feature_overlay = Some(FeatureOverlayState {
            image,
            version,
            overlay_mode: OverlayMode::None,
            tracked_only: true,
            max_features: None,
            min_feature_size: None,
            max_feature_size: None,
            features,
            tree,
        });
    }

    /// Build display feature list for overlay modes (Features/ReprojError/TrackLength).
    fn load_display_features(
        &mut self,
        edited: &EditedReconstruction,
        image: ImageRef,
        version: VersionSerial,
        cached_sift: Option<&CachedSiftFeatures>,
        settings: &FeatureDisplaySettings,
    ) {
        let recon = &*edited.base;
        let img_idx = image.index();
        // Embedded-patches: build features from the inline per-observation
        // keypoints, with affine shapes derived by projecting each point's patch
        // frame. Every embedded observation is tracked (no untracked keypoints),
        // so `tracked_only` is a no-op; size filters and the max-features cap
        // apply just like the SIFT path.
        if recon.feature_indexes().is_none() {
            let mut features = embedded_image_features(edited, img_idx);
            features.retain(|f| {
                let size = feature_size(&f.affine_shape);
                settings.min_feature_size.is_none_or(|mn| size >= mn)
                    && settings.max_feature_size.is_none_or(|mx| size <= mx)
            });
            // Keep the largest features when capping, mirroring the SIFT path
            // (whose cache is pre-sorted by decreasing size).
            if let Some(max) = settings.max_features {
                if features.len() > max {
                    features.sort_by(|a, b| {
                        feature_size(&b.affine_shape).total_cmp(&feature_size(&a.affine_shape))
                    });
                    features.truncate(max);
                }
            }
            populate_feature_diagnostics(&mut features, edited, settings.overlay_mode);
            let tree = build_feature_tree(&features);
            log::info!(
                "Loaded {} embedded features for image {} (mode: {:?})",
                features.len(),
                img_idx,
                settings.overlay_mode,
            );
            self.feature_overlay = Some(FeatureOverlayState {
                image,
                version,
                overlay_mode: settings.overlay_mode,
                tracked_only: settings.tracked_only,
                max_features: settings.max_features,
                min_feature_size: settings.min_feature_size,
                max_feature_size: settings.max_feature_size,
                features,
                tree,
            });
            return;
        }

        let Some(cached) = cached_sift else {
            self.feature_overlay = Some(FeatureOverlayState {
                image,
                version,
                overlay_mode: settings.overlay_mode,
                tracked_only: settings.tracked_only,
                max_features: settings.max_features,
                min_feature_size: settings.min_feature_size,
                max_feature_size: settings.max_feature_size,
                features: Vec::new(),
                tree: PointCloud2::<f32>::new(&[], 0),
            });
            return;
        };

        let feature_to_point = &recon.point_set.image_feature_to_point[img_idx];
        let num_features = cached.positions_xy.len();

        // Apply max_features limit
        let limit = settings
            .max_features
            .map_or(num_features, |m| m.min(num_features));

        // Apply min_feature_size filter: features are sorted by decreasing size,
        // so scan from the end of the prefix to find the cutoff.
        let effective_count = if let Some(min_size) = settings.min_feature_size {
            let mut cutoff = limit;
            for i in (0..limit).rev() {
                if feature_size(&cached.affine_shapes[i]) >= min_size {
                    cutoff = i + 1;
                    break;
                }
                if i == 0 {
                    cutoff = 0;
                }
            }
            cutoff
        } else {
            limit
        };

        let mut features = Vec::with_capacity(effective_count);
        for i in 0..effective_count {
            // Skip features larger than max_feature_size
            if let Some(max_size) = settings.max_feature_size {
                if feature_size(&cached.affine_shapes[i]) > max_size {
                    continue;
                }
            }

            // Through the version: a base row this version deleted leaves the
            // feature untracked rather than pointing at a dead index, and one
            // an edit replaced points at the addition that superseded it.
            let point_index = feature_to_point
                .get(&(i as u32))
                .and_then(|&base| edited.live_index_of_base(base))
                .unwrap_or(UNTRACKED);

            // Skip untracked features if tracked_only is set
            if settings.tracked_only && point_index == UNTRACKED {
                continue;
            }

            features.push(DisplayFeature {
                position: cached.positions_xy[i],
                affine_shape: cached.affine_shapes[i],
                point_index,
                max_track_angle_deg: f32::NAN,
                inverse_depth_z: f32::NAN,
                condition_number: f32::NAN,
            });
        }

        // Populate per-point diagnostics only when the active overlay consumes
        // them. Each iterates a point's observations, so we pay only on demand.
        populate_feature_diagnostics(&mut features, edited, settings.overlay_mode);

        let tree = build_feature_tree(&features);

        let tracked_count = features.iter().filter(|f| f.is_tracked()).count();
        log::info!(
            "Loaded {} features ({} tracked) for image {} (mode: {:?})",
            features.len(),
            tracked_count,
            img_idx,
            settings.overlay_mode,
        );
        self.feature_overlay = Some(FeatureOverlayState {
            image,
            version,
            overlay_mode: settings.overlay_mode,
            tracked_only: settings.tracked_only,
            max_features: settings.max_features,
            min_feature_size: settings.min_feature_size,
            max_feature_size: settings.max_feature_size,
            features,
            tree,
        });
    }

    /// Clear the cached image (e.g., when reconstruction changes).
    #[allow(dead_code)]
    pub fn clear(&mut self) {
        self.loaded_image = None;
        self.feature_overlay = None;
        self.intrinsics.clear();
        self.reset_view();
    }
}

/// Compute the size of a feature from its 2x2 affine shape matrix.
/// Size = average of column norms.
fn feature_size(affine: &[[f32; 2]; 2]) -> f32 {
    let col0_norm = (affine[0][0] * affine[0][0] + affine[1][0] * affine[1][0]).sqrt();
    let col1_norm = (affine[0][1] * affine[0][1] + affine[1][1] * affine[1][1]).sqrt();
    0.5 * (col0_norm + col1_norm)
}

/// Build a 2-D kd-tree over feature positions for hit-testing / hover.
///
/// Indices into the cloud are indices into `features`.
fn build_feature_tree(features: &[DisplayFeature]) -> PointCloud2<f32> {
    let flat: Vec<f32> = features.iter().flat_map(|f| f.position).collect();
    PointCloud2::<f32>::new(&flat, features.len())
}

/// Feature list for an `embedded_patches` reconstruction: every observation
/// landing in `img_idx`, as a tracked feature at its inline keypoint. The affine
/// shape is derived by projecting the point's patch frame into this image
/// (`observation_affine_shape`); it falls back to a degenerate (zero) shape when
/// the point has no usable patch frame, in which case `draw_feature_ellipse`
/// skips the ellipse and only the centre dot draws. O(total observations):
/// embedded recons have no per-image keypoint index (`image_feature_to_point`
/// is empty).
///
/// The walk is over the version's **live** indexes rather than the base's rows:
/// base points less the deleted set, then the additions. A point this version
/// deleted contributes no feature, and one it added or modified contributes the
/// track it holds now.
fn embedded_image_features(edited: &EditedReconstruction, img_idx: usize) -> Vec<DisplayFeature> {
    if !edited.has_keypoints() {
        return Vec::new();
    }
    let mut features = Vec::new();
    for point_idx in edited.live_indexes() {
        let Some(view) = edited.point(point_idx) else {
            continue;
        };
        for (k, obs) in view.observations().iter().enumerate() {
            if obs.image_index as usize == img_idx {
                let Some(position) = view.keypoint_xy(k) else {
                    continue;
                };
                let affine_shape = edited
                    .observation_affine_shape(point_idx, img_idx, position)
                    .unwrap_or([[0.0; 2]; 2]);
                features.push(DisplayFeature {
                    position,
                    affine_shape,
                    point_index: point_idx,
                    max_track_angle_deg: f32::NAN,
                    inverse_depth_z: f32::NAN,
                    condition_number: f32::NAN,
                });
            }
        }
    }
    features
}

/// Populate the per-point diagnostics an overlay mode consumes (only for the
/// modes that need them; each iterates a point's observations, so we pay only
/// on demand).
fn populate_feature_diagnostics(
    features: &mut [DisplayFeature],
    edited: &EditedReconstruction,
    mode: OverlayMode,
) {
    match mode {
        OverlayMode::MaxTrackAngle => {
            for feature in features.iter_mut() {
                if feature.is_tracked() {
                    feature.max_track_angle_deg =
                        compute_max_track_angle_deg(edited, feature.point_index);
                }
            }
        }
        OverlayMode::DepthReliability | OverlayMode::ConditionNumber => {
            for feature in features.iter_mut() {
                if feature.is_tracked() {
                    let Some(view) = edited.point(feature.point_index) else {
                        continue;
                    };
                    let (cond, z) =
                        crate::metrics::compute_point_diagnostics(&edited.base.image_table, &view);
                    feature.condition_number = cond;
                    feature.inverse_depth_z = z;
                }
            }
        }
        _ => {}
    }
}

/// Compute the max pairwise angle (degrees) between world-space rays from
/// observing cameras to a 3D point. Single-observation points return 0.0.
///
/// **A point at infinity is a bearing and every sighting of it casts the same
/// ray**, so its widest pair is zero degrees whatever the baseline. The three
/// stored numbers are that direction rather than a place, and subtracting a
/// camera centre from them would measure the spread of rays to a point one unit
/// from the world origin -- a confident wrong number, and a different one from
/// what Track View reports for the same row.
fn compute_max_track_angle_deg(edited: &EditedReconstruction, point_idx: u32) -> f32 {
    let Some(view) = edited.point(point_idx) else {
        return f32::NAN;
    };
    let point = view.point();
    let observations = view.observations();
    if point.is_at_infinity() {
        return 0.0;
    }
    let point_pos = point.position;
    let mut world_rays: Vec<[f64; 3]> = Vec::with_capacity(observations.len());
    for obs in observations {
        let img_idx = obs.image_index as usize;
        let Some(image) = edited.base.image_table.images.get(img_idx) else {
            continue;
        };
        let cam_center = image.camera_center();
        let dir = point_pos - cam_center;
        let len = (dir.x * dir.x + dir.y * dir.y + dir.z * dir.z).sqrt();
        if len > 1e-12 {
            world_rays.push([dir.x / len, dir.y / len, dir.z / len]);
        }
    }
    if world_rays.len() < 2 {
        return 0.0;
    }
    crate::metrics::compute_max_pairwise_angle(&world_rays)
}
