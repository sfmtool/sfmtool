// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The image menu's `Add Image to Tracks` action: what the viewer adds on top
//! of the core operation.
//!
//! The operation itself, which asks of every point the image does not observe
//! whether the image sees it, where, and whether the sighting agrees with the
//! point's other observations, is
//! [`sfmtool_core::reconstruction::add_image_to_tracks`]. What stays here is the
//! viewer's share: when the entry is greyed and why, the background job that
//! decodes the photographs and runs the operation on a worker, and the Action
//! Log sentence. See `specs/gui/edits/add-image-to-tracks.md`.

use sfmtool_core::reconstruction::add_image_to_tracks::{
    add_image_to_tracks, AddImageToTracksError, AddImageToTracksOptions, AddImageToTracksReport,
};
use sfmtool_core::{EditedReconstruction, SfmrReconstruction};

use crate::action_log::Kind;
use crate::background::{Finished, Job, Operation};
use crate::document::PointMap;
use crate::scene::ImageRef;
use crate::state::AppState;

#[cfg(test)]
pub(crate) mod tests;

/// Why `Add Image to Tracks` is greyed on a node with no points.
pub(crate) const NO_POINTS_HINT: &str =
    "This reconstruction has no points, so there are no tracks to add the image to.";

/// Why it is greyed on a node whose observations are `.sift` feature indexes.
pub(crate) const SIFT_FILES_HINT: &str =
    "This reconstruction's observations are .sift feature indexes, and an added observation \
     has no feature to name. Convert it to embedded patches first.";

/// Why it is greyed on a node whose points carry no patch frame.
pub(crate) const NO_FRAMES_HINT: &str =
    "This reconstruction's points carry no patch frame, and a point's patch is what is looked \
     for in the image.";

/// Why it is greyed on an image with no pose.
pub(crate) const NOT_POSED_HINT: &str =
    "This image is not posed, so no point can be projected into it. Resect Image gives it a \
     pose.";

/// Why it is greyed on an image whose photograph cannot be found.
pub(crate) const NO_PHOTOGRAPH_HINT: &str =
    "This image's photograph is not where the reconstruction says it is, and the image's \
     pixels are what a point is looked for in.";

/// Why the operation cannot run on any image of `edited`, or `None`.
///
/// The node's own reasons, the ones that hold whichever image is asked about:
/// the core operation's refusals, asked before the gesture rather than after.
pub(crate) fn node_refusal(edited: &EditedReconstruction) -> Option<&'static str> {
    if edited.point_count() == 0 {
        return Some(NO_POINTS_HINT);
    }
    if edited.has_feature_indexes() {
        return Some(SIFT_FILES_HINT);
    }
    if !edited.has_patch_frames() {
        return Some(NO_FRAMES_HINT);
    }
    None
}

/// What one call did, as the Action Log says it: `Added frame_22.jpg to 19
/// tracks (363 candidates refused: 249 not in frame, 76 peak at edge, ...)`.
///
/// The refusals are named with their counts, most first, because "why were the
/// others not added" is the question the number raises.
pub(crate) fn outcome_text(name: &str, report: &AddImageToTracksReport) -> String {
    let refused = report.candidates.len() - report.accepted;
    let mut counts = report.refusal_counts();
    counts.sort_by_key(|c| std::cmp::Reverse(c.1));
    let reasons: Vec<String> = counts
        .iter()
        .map(|(refusal, n)| format!("{n} {}", refusal.name().replace('_', " ")))
        .collect();
    let mut text = format!(
        "Added {name} to {} tracks ({refused} candidates refused",
        report.accepted
    );
    if !reasons.is_empty() {
        text.push_str(": ");
        text.push_str(&reasons.join(", "));
    }
    text.push(')');
    text
}

impl AppState {
    /// Why `Add Image to Tracks` cannot run on `image`, or `None` when it can.
    ///
    /// The one sentence the greyed entry's hover, the step and the wire all
    /// refuse with, in the order the menu asks them
    /// ([`crate::image_menu::ImageMenu::add_to_tracks_refusal`]).
    pub(crate) fn add_image_to_tracks_refusal(&self, image: ImageRef) -> Option<String> {
        let Some(menu) = self.image_menu(image.recon) else {
            return Some(crate::state::NOT_LOADED.to_string());
        };
        menu.add_to_tracks_refusal(image.index())
            .map(str::to_string)
    }

    /// Start adding `image` to the tracks of its node on a worker thread.
    ///
    /// A bulk edit that renumbers nothing: the version it lands as holds every
    /// point at its index, with the accepted observations added. Returns as
    /// soon as the worker is running, and **nothing is logged here**: the entry
    /// is the outcome's, written on the frame the answer lands. An `Err` is a
    /// refusal to begin, logged the way the synchronous edits log theirs.
    pub(crate) fn start_add_image_to_tracks(&mut self, image: ImageRef) -> Result<(), String> {
        let outcome = match self.add_image_to_tracks_job(image) {
            Ok(job) => self.start_background_task(Operation::ADD_IMAGE_TO_TRACKS, image.recon, job),
            Err(message) => Err(message),
        };
        if let Err(message) = &outcome {
            self.action_log.fail(Kind::Edit, message.clone());
        }
        outcome
    }

    /// The step itself, as a function of the
    /// [`Progress`](sfmtool_core::progress::Progress) it reports through.
    ///
    /// What crosses to the worker is a clone of the value at the cursor, whose
    /// `base` is the `Arc` the node goes on drawing, and the photographs'
    /// sources: the pyramids the viewer already holds and the paths of the
    /// rest, which the worker reads. A photograph that cannot be read leaves
    /// that image out of every point's references rather than refusing the
    /// step; the target's own was checked for before the step began.
    pub(crate) fn add_image_to_tracks_job(&self, image: ImageRef) -> Result<Job, String> {
        let id = image.recon;
        let node = self
            .node(id)
            .ok_or_else(|| crate::state::NOT_LOADED.to_string())?;
        let label = node.label.clone();
        let name = node
            .recon()
            .image_table
            .images
            .get(image.index())
            .map(|im| crate::resect::basename(&im.name).to_string())
            .ok_or_else(|| format!("{label} has no image {}.", image.index()))?;
        if let Some(why) = self.add_image_to_tracks_refusal(image) {
            return Err(format!("Add {name} to tracks in {label} refused: {why}"));
        }
        let edited = node.history.current().clone();
        let every: Vec<usize> = (0..node.recon().image_count()).collect();
        let sources = self.view_sources_for(id, &every)?;
        let index = image.index();
        Ok(Box::new(move |progress| {
            let refuse = |why: String| format!("Add {name} to tracks in {label} refused: {why}");
            let [decode, work] = progress.split([1.0, 1.0]);
            let Ok(views) = sources.decode_available(&decode) else {
                return Finished::Cancelled;
            };

            // Materialise only when there is an overlay to fold in, as the
            // other bulk edits do.
            let (materialised, mat_map) =
                if edited.deleted_points.is_empty() && edited.added.points.is_empty() {
                    (None, None)
                } else {
                    let _phase = work.phase("materialise");
                    let (value, map) = edited.materialize();
                    (Some(value), Some(PointMap::Rows(map)))
                };
            let source: &SfmrReconstruction = match materialised.as_ref() {
                Some(value) => value,
                None => &edited.base,
            };

            let slots = views.pyramid_slots();
            let (next, report) = match add_image_to_tracks(
                source,
                index,
                &slots,
                &AddImageToTracksOptions::default(),
                &work,
            ) {
                Ok(added) => added,
                Err(AddImageToTracksError::Cancelled) => return Finished::Cancelled,
                Err(e) => return Finished::Failed(refuse(e.to_string())),
            };
            let text = outcome_text(&name, &report);
            if report.accepted == 0 {
                return Finished::NoChange(text);
            }
            // Every point keeps its index: the identity, stated.
            let mut steps = Vec::new();
            steps.extend(mat_map);
            steps.push(PointMap::Removed(Vec::new()));
            Finished::Produced {
                value: next,
                map: PointMap::Chain(steps),
                version_label: format!("Added {name} to tracks"),
                text,
            }
        }))
    }
}
