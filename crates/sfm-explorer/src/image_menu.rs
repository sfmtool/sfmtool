// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The image menu: the context menu of one image of a reconstruction.
//!
//! One menu shown in two places, a Scene tree image row (under a
//! reconstruction's *Camera Images* group) and a thumbnail of the Image
//! Browser strip. Both call [`show`], so the entries, their order, their greyed
//! states and their hover reasons are the same by construction, and both hand
//! the chosen [`ImageMenuAction`] to the dock, which carries it out through one
//! path. See `specs/gui/scene-graph.md` and
//! `specs/gui/multi-panel-image-browser.md`.
//!
//! What a place adds is where it records the entries it laid out (the tree's
//! row ids, the strip's own ids), which is what its headless tests aim clicks
//! at.

use eframe::egui;

use crate::scene::{ImageRef, ReconId};
use crate::state::AppState;
use sfmtool_core::geometry::MIN_OTHER_POSED_IMAGES;

#[cfg(test)]
mod tests;

/// An entry of the image menu, chosen on one image.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ImageMenuAction {
    /// `Resect Image`: re-estimate the image's pose against structure held out
    /// from it, as the node's next version.
    Resect,
    /// `Add Image to Tracks`: add the image's observations of the points it
    /// sees and does not observe, as the node's next version.
    AddToTracks,
    /// `Move Camera`: look through the image and take its camera in hand.
    MoveCamera,
    /// `Delete Image`: remove the image, as the node's next version.
    Delete,
}

/// The menu's entries in the order it shows them: the key each place records
/// an entry's rect under, and its label.
pub(crate) const ENTRIES: [(&str, &str); 4] = [
    (RESECT, "Resect Image"),
    (ADD_TO_TRACKS, "Add Image to Tracks"),
    (MOVE_CAMERA, "Move Camera"),
    (DELETE_IMAGE, "Delete Image"),
];

/// The key of the `Resect Image` entry.
pub(crate) const RESECT: &str = "resect";
/// The key of the `Add Image to Tracks` entry.
pub(crate) const ADD_TO_TRACKS: &str = "add_to_tracks";
/// The key of the `Move Camera` entry.
pub(crate) const MOVE_CAMERA: &str = "move_camera";
/// The key of the `Delete Image` entry.
pub(crate) const DELETE_IMAGE: &str = "delete_image";

/// What the image menu needs to know about the node its image belongs to,
/// computed once per node and frame rather than once per image.
#[derive(Debug, Clone)]
pub(crate) struct ImageMenu {
    /// Whether each image carries a pose at all. A `.sfmr` row always has the
    /// fields; a non-finite one is a placeholder rather than a registration.
    posed: Vec<bool>,
    /// How many images of the node are posed.
    posed_count: usize,
    /// Why the node's cluster-patches file will not do for a resection, or
    /// `None` when it is current.
    cluster_patches: Option<String>,
    /// Why `Add Image to Tracks` cannot run on any image of the node, or
    /// `None`: an operation running on it, or the node's own reasons
    /// ([`crate::add_image_to_tracks::node_refusal`]).
    add_to_tracks: Option<String>,
    /// Per image, where its photograph would come from.
    ///
    /// A path is looked at lazily, by [`Self::add_to_tracks_refusal`], so a
    /// frame that shows no menu reads nothing from the file system.
    photographs: Vec<Photograph>,
}

/// Where one image's pixels would come from, for the menu's gate.
#[derive(Debug, Clone)]
enum Photograph {
    /// Already decoded in the viewer's cache.
    Cached,
    /// To be read from this path.
    File(std::path::PathBuf),
}

impl ImageMenu {
    /// Why `Resect Image` is unavailable for image `index`, or `None` when it
    /// is available.
    ///
    /// The image's own reasons first, since they hold whatever the node's
    /// files say: an image with no pose, then too few other posed images, then
    /// the cluster-patches file.
    pub(crate) fn resect_refusal(&self, index: usize) -> Option<&str> {
        if !self.posed.get(index).copied().unwrap_or(false) {
            return Some(NOT_POSED_HINT);
        }
        // The target itself is one of the posed images, so "three others" is
        // four in total.
        if self.posed_count < MIN_OTHER_POSED_IMAGES + 1 {
            return Some(TOO_FEW_POSED_HINT);
        }
        self.cluster_patches.as_deref()
    }

    /// Why `Add Image to Tracks` is unavailable for image `index`, or `None`
    /// when it is available.
    ///
    /// The image's pose first, then the node's reasons, then the photograph,
    /// the one reason that costs a look at the file system.
    pub(crate) fn add_to_tracks_refusal(&self, index: usize) -> Option<&str> {
        if !self.posed.get(index).copied().unwrap_or(false) {
            return Some(crate::add_image_to_tracks::NOT_POSED_HINT);
        }
        if let Some(why) = self.add_to_tracks.as_deref() {
            return Some(why);
        }
        match self.photographs.get(index) {
            Some(Photograph::Cached) => None,
            Some(Photograph::File(path)) if path.is_file() => None,
            _ => Some(crate::add_image_to_tracks::NO_PHOTOGRAPH_HINT),
        }
    }
}

/// Why `Resect Image` is greyed on an image with no pose.
pub(crate) const NOT_POSED_HINT: &str =
    "This image is not posed, so there is no pose to re-estimate against the rest.";

/// Why `Resect Image` is greyed on a node with too little of a reconstruction
/// to hold anything out from.
pub(crate) const TOO_FEW_POSED_HINT: &str =
    "Fewer than three other images of this reconstruction are posed. Two cameras fix \
     structure only up to their own degenerate freedoms, so re-estimating a pose \
     against them would measure the pair rather than the scene.";

impl AppState {
    /// The image menu's view of node `id` as it stands, or `None` when the node
    /// is not loaded.
    ///
    /// Reads the cluster-patches state without refreshing it: the Scene tree
    /// refreshes every node's index files once per frame before it asks, and
    /// the step refreshes before it asks again.
    pub(crate) fn image_menu(&self, id: ReconId) -> Option<ImageMenu> {
        let node = self.node(id)?;
        let posed: Vec<bool> = node
            .recon()
            .image_table
            .images
            .iter()
            .map(|image| {
                image.quaternion_wxyz.coords.iter().all(|c| c.is_finite())
                    && image.translation_xyz.iter().all(|c| c.is_finite())
            })
            .collect();
        let recon = node.recon();
        let photographs = recon
            .image_table
            .images
            .iter()
            .enumerate()
            .map(|(i, image)| {
                let cached = self
                    .full_res_cache
                    .get(&ImageRef::new(id, i))
                    .is_some_and(|slot| slot.is_some());
                if cached {
                    Photograph::Cached
                } else {
                    Photograph::File(recon.workspace_dir.join(&image.name))
                }
            })
            .collect();
        let add_to_tracks = self.busy_refusal(id).or_else(|| {
            crate::add_image_to_tracks::node_refusal(node.history.current()).map(str::to_string)
        });
        Some(ImageMenu {
            posed_count: posed.iter().filter(|&&p| p).count(),
            posed,
            cluster_patches: self.resect_cluster_patches_refusal(id),
            add_to_tracks,
            photographs,
        })
    }

    /// Why `Resect Image` cannot run on `image`, or `None` when it can.
    ///
    /// The one sentence the greyed entry's hover, the step and the wire all
    /// refuse with.
    pub(crate) fn resect_image_refusal(&self, image: ImageRef) -> Option<String> {
        let Some(menu) = self.image_menu(image.recon) else {
            return Some("That reconstruction is no longer loaded.".to_string());
        };
        menu.resect_refusal(image.index()).map(str::to_string)
    }
}

/// Lay out the image menu for image `index` and give back the entry chosen,
/// if one was.
///
/// `mark` is called with each entry's key (one of [`ENTRIES`]) and its
/// response, in the order the entries are shown, so the place showing the menu
/// can record where each landed. The menu is closed when an entry is chosen.
///
/// Every entry stays visible and is greyed rather than hidden when it is
/// unavailable: the action exists on every image, and an entry that vanishes
/// reads as an action that was never implemented. The hover text says which
/// reason applies.
pub(crate) fn show(
    ui: &mut egui::Ui,
    index: usize,
    menu: &ImageMenu,
    mark: &mut dyn FnMut(&'static str, &egui::Response),
) -> Option<ImageMenuAction> {
    let mut chosen = None;

    let refusal = menu.resect_refusal(index);
    let resect = ui
        .add_enabled(refusal.is_none(), egui::Button::new(ENTRIES[0].1))
        .on_disabled_hover_text(refusal.unwrap_or_default())
        .on_hover_text(
            "Re-estimate this image's pose against structure re-triangulated without it, \
             from the tracks and the clusters of the cluster patches file, and keep the \
             answer as a version of this reconstruction. Undo (Ctrl+Z) puts the stored \
             pose back.",
        );
    mark(RESECT, &resect);
    if resect.clicked() {
        chosen = Some(ImageMenuAction::Resect);
    }

    // Beside the resection, because it is what a re-posed image wants next:
    // the pose is new, and the tracks it can now see are not yet its own.
    let refusal = menu.add_to_tracks_refusal(index);
    let add = ui
        .add_enabled(refusal.is_none(), egui::Button::new(ENTRIES[1].1))
        .on_disabled_hover_text(refusal.unwrap_or_default())
        .on_hover_text(
            "Look for every point this image does not observe in its photograph, and add \
             the observations whose appearance agrees with the point's other observations, \
             as a version of this reconstruction. Nothing else moves. Runs in the \
             background; Undo (Ctrl+Z) takes the observations back out.",
        );
    mark(ADD_TO_TRACKS, &add);
    if add.clicked() {
        chosen = Some(ImageMenuAction::AddToTracks);
    }

    ui.separator();
    // The hand, beside the estimator: where a resection re-computes a pose
    // from correspondences, this hands the camera to the reviewer. It enters
    // camera view first, because the lock *is* camera view with the camera
    // coming along.
    let move_camera = ui.add(egui::Button::new(ENTRIES[2].1)).on_hover_text(
        "Look through this image and take its camera in hand: every navigation \
         input moves it, and M or Enter keeps the pose as a version of this \
         reconstruction.",
    );
    mark(MOVE_CAMERA, &move_camera);
    if move_camera.clicked() {
        chosen = Some(ImageMenuAction::MoveCamera);
    }

    ui.separator();
    // No confirmation: this is an edit with a history behind it, and Undo is
    // the answer to a mis-click, as it is for the entries above.
    let delete = ui.add(egui::Button::new(ENTRIES[3].1)).on_hover_text(
        "Remove this image from the reconstruction, with its observations and any \
         track left with none. Undo (Ctrl+Z) puts it back.",
    );
    mark(DELETE_IMAGE, &delete);
    if delete.clicked() {
        chosen = Some(ImageMenuAction::Delete);
    }

    if chosen.is_some() {
        ui.close();
    }
    chosen
}
