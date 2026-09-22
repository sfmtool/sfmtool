// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! [`AppState`]'s edits: the operations that give a node a new version, and the
//! three that move its cursor -- undo, redo, and the Edit History panel's jump,
//! which is the two of them repeated.
//!
//! See `specs/gui/document-model.md` and `specs/gui/edit-history.md`. Each edit
//! is a function from the value at the node's cursor to the next value, pushed
//! onto that node's history with the map that says what it did to point
//! indexes. Nothing here mutates a reconstruction in place: a point edit writes
//! the *overlay* the new version owns, and a bulk edit builds a whole new base
//! out of the old one.
//!
//! They are here together because two of them are the two shapes every other
//! one follows:
//!
//! - [`AppState::delete_selected_point`] is the **point edit**. It calls
//!   `EditedReconstruction::delete_point` on a clone of the current value, so
//!   the base is the same `Arc` and nothing but the deleted set changed.
//! - [`AppState::delete_image`] is the **bulk edit**. It materialises the
//!   current value when the overlay is not empty, runs
//!   `SfmrReconstruction::subset_by_image_indices` over it, and the output is
//!   the next version's base with an empty overlay, under the row map
//!   `RowMap::by_scan` reads off that call's input and output.

use std::sync::Arc;
use std::time::Instant;

use sfmtool_core::camera::remap::{ImageU8, ImageU8Pyramid};
use sfmtool_core::camera::CameraIntrinsics;
use sfmtool_core::geometry::RigidTransform;
use sfmtool_core::patch::cloud::{PatchExtent, PatchNormal, ViewReduce};
use sfmtool_core::patch::normal_refine::ProjectedImage;
use sfmtool_core::progress_note;
use sfmtool_core::reconstruction::prune_covered::{
    prune_covered_observations, PruneCoveredError, PruneCoveredOptions, PruneCoveredReport,
};
use sfmtool_core::reconstruction::triangulation::{
    retriangulate_points, RetriangulateError, RetriangulateOptions, RetriangulateReport,
    RetriangulateWhich,
};
use sfmtool_core::{EditedReconstruction, RowMap, SfmrReconstruction};

use crate::action_log::Kind;
use crate::background::{Finished, Job, Operation};
use crate::document::{PointMap, VersionSerial};
use crate::progress::Collector;
use crate::resect::ResectFrom;
use crate::scene::{ImageRef, PointRef, ReconId};

use super::{AppState, PYRAMID_LEVELS};

/// The patch-frame normal the viewer's `sift_files` → `embedded_patches`
/// conversion seeds each point with: the mean of its point-to-camera
/// directions, which is `sfm xform --to-embedded-patches`'s own default and the
/// only seed that needs nothing but the solve.
const CONVERSION_NORMAL: PatchNormal = PatchNormal::MeanViewing;

/// The half-extent policy that conversion sizes each patch by, and the CLI's
/// own default: `2.5 ×` the median projected keypoint scale across the point's
/// views, so the full patch edge is five times the feature. `PatchExtent`'s
/// `Default` *is* that pair, and it is spelled out here rather than taken from
/// `Default` so the two layers are one written-down number rather than two that
/// happen to agree.
const CONVERSION_EXTENT: PatchExtent = PatchExtent::FeatureSize {
    factor: 2.5,
    across: ViewReduce::Median,
};

/// Why `node` cannot be converted to `embedded_patches`, or `None` when it can.
///
/// A free function over the node and the busy sentence rather than a method,
/// because the two callers reach it from different sides: the Scene tree's menu
/// is drawn inside a walk that holds the node and has already given up its
/// borrow of the state, and [`AppState::convert_to_embedded_patches_refusal`]
/// holds the state and looks the node up. One definition, so the greyed entry's
/// hover text and the refusal a call gets are the same sentence.
///
/// `busy` is [`AppState::busy_refusal`]'s answer for this node, and it comes
/// first: an operation already running on the node is the reason that will
/// still be true a moment later.
pub(crate) fn convert_refusal(
    node: &crate::scene::SceneNode,
    busy: Option<&str>,
) -> Option<String> {
    if let Some(why) = busy {
        return Some(why.to_string());
    }
    // The observation source, not `SceneNode::has_patch_data`: that answers
    // whether the node carries reference *bitmaps* to texture its surfels
    // with, which this conversion does not produce and which the surfel
    // renderer is the one reader of.
    node.recon().point_set.feature_indexes().is_none().then(|| {
        format!(
            "{} is already an embedded_patches reconstruction.",
            node.label
        )
    })
}

/// Why `node` cannot be retriangulated, or `None` when it can.
///
/// The three reasons a caller can see without solving anything, and they are
/// the core operation's own refusals ([`RetriangulateError`]) asked before the
/// gesture rather than after it: the observations carry no pixel to cast a ray
/// through, no image carries a pose, and the posed images do not share one
/// lens.
///
/// A free function over the node and the busy sentence for the reason
/// [`convert_refusal`] is one: the Scene tree's menu is drawn inside a walk
/// that has already given up its borrow of the state, and the edits look the
/// node up. One definition, so the greyed entry's hover text and the refusal a
/// call gets are the same sentence.
///
/// `busy` is [`AppState::busy_refusal`]'s answer for this node, and it comes
/// first: an operation already running on the node is the reason that will
/// still be true a moment later.
pub(crate) fn retriangulate_refusal(
    node: &crate::scene::SceneNode,
    busy: Option<&str>,
) -> Option<String> {
    if let Some(why) = busy {
        return Some(why.to_string());
    }
    let edited = node.history.current();
    if !edited.has_keypoints() {
        return Some(
            "This reconstruction's observations are .sift feature indexes with no inline \
             keypoints, and retriangulation needs a pixel per observation."
                .to_string(),
        );
    }
    match edited.posed_lens_count() {
        0 => Some("No image of this reconstruction carries a pose.".to_string()),
        1 => None,
        n => Some(format!(
            "Retriangulation reads one shared camera, and these images are taken through {n}."
        )),
    }
}

/// Why `node`'s covered observations cannot be pruned, or `None` when they can.
///
/// The three reasons a caller can see without reading a single footprint, and
/// they are the core operation's own refusals ([`PruneCoveredError`]) asked
/// before the gesture rather than after it: the points carry no patch frame to
/// read a footprint off, the observations carry no pixel for one to sit at, and
/// no image carries a pose to project through.
///
/// A free function over the node and the busy sentence for the reason
/// [`retriangulate_refusal`] is one, and `busy` comes first for the same reason:
/// an operation already running on the node is the reason that will still be
/// true a moment later.
pub(crate) fn prune_covered_refusal(
    node: &crate::scene::SceneNode,
    busy: Option<&str>,
) -> Option<String> {
    if let Some(why) = busy {
        return Some(why.to_string());
    }
    let edited = node.history.current();
    if !edited.has_patch_frames() {
        return Some(
            "This reconstruction's points carry no patch frame, and an observation's \
             footprint is that frame projected into the image that saw it."
                .to_string(),
        );
    }
    if !edited.has_keypoints() {
        return Some(
            "This reconstruction's observations are .sift feature indexes with no inline \
             keypoints, and a footprint needs a pixel to sit at."
                .to_string(),
        );
    }
    (edited.posed_lens_count() == 0)
        .then(|| "No image of this reconstruction carries a pose.".to_string())
}

/// What one prune did, as the Action Log says it.
fn prune_covered_summary(report: &PruneCoveredReport) -> String {
    if !report.changed {
        return "no effect, no observation is covered by a finer one".to_string();
    }
    let mut text = format!(
        "{} of {} observations retired",
        report.census.rows_removed, report.observations_before
    );
    let dropped = report.points_before - report.points_after;
    if dropped > 0 {
        text.push_str(&format!(", {dropped} points dropped"));
    }
    if report.protected_rows_spared > 0 {
        text.push_str(&format!(
            ", {} pinned rows spared",
            report.protected_rows_spared
        ));
    }
    if report.degenerate_rows > 0 {
        text.push_str(&format!(
            ", {} rows without a usable footprint",
            report.degenerate_rows
        ));
    }
    text
}

/// What one retriangulation did, as the Action Log says it.
///
/// The version label is the short half the Edit History panel lists; the
/// sentence is that label plus the numbers, which is the shape every other edit
/// here reports in.
fn retriangulate_summary(report: &RetriangulateReport) -> String {
    let mut text = format!("{} of {} points moved", report.moved, report.read);
    if report.crossed > 0 {
        text.push_str(&format!(", {} crossed to or from infinity", report.crossed));
    }
    if report.kept > 0 {
        text.push_str(&format!(", {} too thinly seen to place", report.kept));
    }
    if report.held > 0 {
        text.push_str(&format!(", {} held", report.held));
    }
    if report.median_shift.is_finite() {
        text.push_str(&format!(", median shift {:.4}", report.median_shift));
    }
    text
}

/// Decoded images for one edit, owning what a [`ProjectedImage`] borrows.
///
/// One entry per image of the node, because the patch kernels index their view
/// slice by image index; the entries the call does not read are one-pixel
/// placeholders.
pub(crate) struct DecodedViews {
    cameras: Vec<CameraIntrinsics>,
    poses: Vec<RigidTransform>,
    pyramids: Vec<Arc<ImageU8Pyramid>>,
}

/// Where one photometric step's photographs are to come from, before any of
/// them has been decoded.
///
/// What crosses to a **worker**: the poses and the cameras, which are cheap,
/// and per image either the pyramid the viewer already holds -- shared rather
/// than copied, which is what the cache's `Arc` is for -- or the path to read
/// the photograph from. So a step over photographs the cache has costs the GUI
/// thread a handful of `Arc` clones and the worker nothing at all, and the file
/// reads and the pyramid builds the rest of them need happen where every other
/// second of that step already happens.
///
/// A photograph the worker reads is dropped with the task rather than put in
/// the cache: the cache is `AppState`'s and the worker cannot reach it, and the
/// panels fill it for what they draw.
pub(crate) struct ViewSources {
    cameras: Vec<CameraIntrinsics>,
    poses: Vec<RigidTransform>,
    sources: Vec<ViewSource>,
}

/// Where one image's pixels come from.
enum ViewSource {
    /// Already decoded and pyramided, shared with whatever else holds it.
    Decoded(Arc<ImageU8Pyramid>),
    /// To be read from this path, with the image's name for the refusal.
    Read(std::path::PathBuf, String),
    /// An image the step does not read: a one-pixel placeholder, which no
    /// kernel samples.
    Unused,
}

impl ViewSources {
    /// Decode what has not been decoded and build the pyramids it needs,
    /// reporting through `progress`.
    ///
    /// A photograph the cache had arrives here already pyramided, so all this
    /// does with it is keep the `Arc`; what is built is one pyramid per
    /// photograph read from disk, and the one-pixel placeholder, which every
    /// unused slot shares.
    ///
    /// The refusal a file that cannot be read produces is the caller's own
    /// sentence, arriving from the worker rather than from the gesture: the
    /// step cannot know before it starts which photographs are readable, and a
    /// step that answered that question first would be the wait this type
    /// exists to remove.
    pub(crate) fn decode(
        self,
        progress: &sfmtool_core::progress::Progress<'_>,
    ) -> Result<DecodedViews, String> {
        let mut phase = progress.phase("decode images");
        let placeholder = Arc::new(ImageU8Pyramid::from_image(
            ImageU8::new(1, 1, 3, vec![0u8; 3]),
            PYRAMID_LEVELS,
        ));
        let mut pyramids = Vec::with_capacity(self.sources.len());
        let mut read = 0usize;
        let mut reused = 0usize;
        for source in self.sources {
            pyramids.push(match source {
                ViewSource::Decoded(pyramid) => {
                    reused += 1;
                    pyramid
                }
                ViewSource::Read(path, name) => {
                    read += 1;
                    let image = crate::state::decode_full_res(&path)
                        .ok_or(format!("Cannot read {name}."))?;
                    Arc::new(ImageU8Pyramid::from_image(image, PYRAMID_LEVELS))
                }
                ViewSource::Unused => Arc::clone(&placeholder),
            });
        }
        progress_note!(
            phase,
            "{read} read from disk, {reused} reused from the cache"
        );
        Ok(DecodedViews {
            cameras: self.cameras,
            poses: self.poses,
            pyramids,
        })
    }
}

impl DecodedViews {
    /// The borrowed form a patch kernel takes.
    pub(crate) fn views(&self) -> Vec<ProjectedImage<'_>> {
        self.cameras
            .iter()
            .zip(&self.poses)
            .zip(&self.pyramids)
            .map(|((camera, cam_from_world), pyramid)| ProjectedImage {
                camera,
                cam_from_world,
                pyramid: pyramid.as_ref(),
            })
            .collect()
    }
}

#[cfg(test)]
pub(crate) mod tests;

/// What a panel's gesture on a point asked of the app.
///
/// Reported rather than done, for the reason every panel reports its gestures:
/// the panel holds the node borrowed out of [`AppState`] while it draws, and
/// each of these needs that state mutably. Two panels report it -- the 3D
/// viewport's point context menu and its double-click
/// ([`crate::viewer_3d::Viewer3D`]), and the Image Detail panel's feature menu
/// and its double-click ([`crate::image_detail::ImageDetail`]) -- so one
/// gesture is one code path wherever it was made.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PointGesture {
    /// A menu opened on this point. Select it, so that what the entries will
    /// act on is also what the rest of the viewer is looking at.
    Opened(PointRef),
    /// `Edit on Bench` was chosen, or the point was double-clicked: put its
    /// track on the bench and show Track View.
    EditOnBench(PointRef),
    /// `Retriangulate Point` was chosen: re-solve this point from its own
    /// observations.
    Retriangulate(PointRef),
}

impl AppState {
    /// Delete the selected point from the node it belongs to.
    ///
    /// A point edit: the version's overlay gains one deleted index, the base is
    /// untouched, and every other index still means what it meant. Returns the
    /// message the caller reports, or `Err` when there is nothing to delete.
    pub fn delete_selected_point(&mut self) -> Result<(), String> {
        let point = self
            .selected_point
            .ok_or_else(|| "No point is selected.".to_string())?;
        self.delete_point(point)
    }

    /// Delete one point, named by ref.
    pub fn delete_point(&mut self, point: PointRef) -> Result<(), String> {
        if let Some(why) = self.busy_refusal(point.recon) {
            return Err(why);
        }
        let index = self
            .scene
            .iter()
            .position(|n| n.id == point.recon)
            .ok_or_else(|| "That reconstruction is no longer loaded.".to_string())?;
        let node = &mut self.scene[index];
        let mut next = node.history.current().clone();
        next.delete_point(point.point)
            .map_err(|e| format!("Cannot delete that point: {e}"))?;
        let label = node.label.clone();
        let text = format!("Deleted point {} in {label}", point.point);
        let serial = node
            .history
            .push(next, PointMap::Removed(vec![point.point]), text.clone());
        let parent = version_before(node, serial);
        self.follow_selection_forward(point.recon);
        self.action_log
            .record(Kind::Edit, format!("{text} ({parent} → {serial})"));
        Ok(())
    }

    /// Why retriangulating `id` is refused right now, or `None`.
    ///
    /// [`retriangulate_refusal`]'s question asked of a node this state holds,
    /// which is the form the wire and the job want; the Scene tree's menu asks
    /// the free function directly, because the walk that draws it has the node
    /// and not the state.
    pub(crate) fn retriangulate_refusal(&self, id: ReconId) -> Option<String> {
        let busy = self.busy_refusal(id);
        retriangulate_refusal(self.node(id)?, busy.as_deref())
    }

    /// Carry out whatever a panel's gesture on a point asked for.
    ///
    /// Here rather than in the panel for the reason every other panel response
    /// is applied outside it: the panel holds the node borrowed out of this
    /// state while it draws, and each of these needs it mutably.
    ///
    /// The selection moves first in every case, so a menu that merely *opened*
    /// on a point leaves the rest of the viewer looking at that point whether
    /// or not an entry is chosen afterwards.
    pub fn apply_point_gesture(&mut self, request: PointGesture) {
        let point = match request {
            PointGesture::Opened(point)
            | PointGesture::EditOnBench(point)
            | PointGesture::Retriangulate(point) => point,
        };
        self.select_point(point);
        match request {
            PointGesture::Opened(_) => {}
            PointGesture::EditOnBench(point) => {
                // The same call ticking Track View's Edit box makes. That
                // box lives inside the panel and so has nothing to raise;
                // this one is reached from the viewport, and a track staged
                // into a panel nobody can see is a gesture with no answer.
                match self.put_point_on_bench(point) {
                    Ok(_) => self.show_panel(crate::dock::Tab::TrackView),
                    Err(why) => self.action_log.fail(Kind::Bench, why),
                }
            }
            PointGesture::Retriangulate(point) => {
                let _ = self.retriangulate_point(point);
            }
        }
    }

    /// Re-solve one point from its own observations, at the poses and the lens
    /// its reconstruction already holds, and install the answer as that node's
    /// next version.
    ///
    /// A point edit: the new version's overlay carries the re-solved record and
    /// the base is the same `Arc`, so every index but this one still means what
    /// it meant. Delete-and-re-add gives the point a **new index**, which is why
    /// the map is a [`PointMap::Replaced`] rather than the empty one a deletion
    /// pushes -- a selection, a copied id and a panel's prepared state all
    /// follow the point through it.
    ///
    /// Nothing else moves: no camera, no lens, and no other point. What this
    /// point's observations support at this geometry is the whole of what it
    /// decides, and the verdict the core operation reached is in the sentence
    /// the Action Log keeps.
    ///
    /// See `specs/gui/edits/retriangulate-point.md`.
    pub fn retriangulate_point(&mut self, point: PointRef) -> Result<(), String> {
        let started = Instant::now();
        // The level the Action Log toolbar's checkbox last left, read as the
        // operation starts so that a change to it takes effect on the next one.
        let collector = Collector::new(self.action_log.detailed_timing());
        match self.retriangulate_point_inner(point, &collector) {
            Ok(message) => {
                self.action_log
                    .record_done(Kind::Edit, started, message, collector.take());
                Ok(())
            }
            Err(message) => {
                self.action_log.fail(Kind::Edit, message.clone());
                Err(message)
            }
        }
    }

    /// The edit itself: `Ok` carries the Action Log's sentence, `Err` the
    /// refusal's.
    fn retriangulate_point_inner(
        &mut self,
        point: PointRef,
        collector: &Collector,
    ) -> Result<String, String> {
        if let Some(why) = self.busy_refusal(point.recon) {
            return Err(why);
        }
        let index = self
            .scene
            .iter()
            .position(|n| n.id == point.recon)
            .ok_or_else(|| "That reconstruction is no longer loaded.".to_string())?;
        let label = self.scene[index].label.clone();
        let refuse = |why: String| format!("Cannot retriangulate point {}: {why}", point.point);
        if let Some(why) = retriangulate_refusal(&self.scene[index], None) {
            return Err(refuse(why));
        }

        let edited = self.scene[index].history.current();
        let (next, map, report) = {
            let _phase = collector.phase("retriangulate");
            retriangulate_points(
                edited,
                RetriangulateWhich::These(&[point.point]),
                &RetriangulateOptions::default(),
                &collector.progress(),
            )
            .map_err(|e| refuse(e.to_string()))?
        };

        // One point read, so the census names one verdict, and that verdict is
        // what a reader wants to be told about this gesture.
        let verdict = report
            .census
            .sole_verdict()
            .map_or_else(String::new, |verdict| format!(": {}", verdict.label()));
        let text = format!("Retriangulated point {} in {label}", point.point);
        let node = &mut self.scene[index];
        let serial = {
            let _phase = collector.phase("push version");
            node.history.push(next, map, text.clone())
        };
        let parent = version_before(node, serial);
        self.follow_selection_forward(point.recon);
        Ok(format!("{text}{verdict} ({parent} → {serial})"))
    }

    /// Start a retriangulation of every point of `id` on a worker thread.
    ///
    /// A bulk edit: every point's geometry is re-read from its own observations
    /// at the poses and the lens the value already holds, so the next version is
    /// a whole new base. No point is deleted and none is created, so the indexes
    /// do not move; what the panels cached *about* the geometry is the caller's
    /// to drop, as it is after an adjustment.
    ///
    /// Returns as soon as the worker is running, and **nothing is logged
    /// here**: the entry is the outcome's, written by
    /// [`AppState::poll_background_task`] on the frame the answer lands, from
    /// the instant the operation started and in the name of whoever asked for
    /// it. An `Err` is a refusal to *begin*, logged the way the synchronous
    /// edits log theirs.
    pub fn start_retriangulate_all_points(&mut self, id: ReconId) -> Result<(), String> {
        let outcome = match self.retriangulate_all_points_job(id) {
            Ok(job) => self.start_background_task(Operation::RETRIANGULATE_ALL_POINTS, id, job),
            Err(message) => Err(message),
        };
        if let Err(message) = &outcome {
            self.action_log.fail(Kind::Edit, message.clone());
        }
        outcome
    }

    /// The retriangulation itself, as a function of the
    /// [`Progress`](sfmtool_core::progress::Progress) it reports through.
    ///
    /// The closure owns what it reads, the way the adjustment's does: the
    /// overlay fold and the solve are pure functions of the value at the cursor,
    /// so what crosses to the worker is a clone of [`EditedReconstruction`]
    /// whose `base` is the very `Arc` the node goes on drawing.
    pub(crate) fn retriangulate_all_points_job(&self, id: ReconId) -> Result<Job, String> {
        let label = self
            .node(id)
            .map(|node| node.label.clone())
            .ok_or_else(|| "That reconstruction is no longer loaded.".to_string())?;
        // The gate is the menu entry's own, so the entry and the edit cannot
        // disagree about when the retriangulation can run.
        if let Some(why) = self.retriangulate_refusal(id) {
            return Err(format!(
                "Retriangulate all points of {label} refused: {why}"
            ));
        }
        let edited = self
            .node(id)
            .expect("just resolved")
            .history
            .current()
            .clone();
        Ok(Box::new(move |progress| {
            let refuse =
                |why: String| format!("Retriangulate all points of {label} refused: {why}");

            // The core operation's own three stages nest directly under the
            // operation's, since this `Progress` is at the top of it, and the
            // overlay fold it does is one of them.
            let (next, map, report) = match retriangulate_points(
                &edited,
                RetriangulateWhich::All,
                &RetriangulateOptions::default(),
                progress,
            ) {
                Ok(solved) => solved,
                // The one error that is not a refusal: the operation was asked
                // to stop and did, which the log words as a cancellation rather
                // than as a failure of the solve.
                Err(RetriangulateError::Cancelled) => return Finished::Cancelled,
                Err(e) => return Finished::Failed(refuse(e.to_string())),
            };

            let version_label = format!("Retriangulated {label}");
            let text = format!("{version_label}: {}", retriangulate_summary(&report));
            Finished::Produced {
                // `RetriangulateWhich::All` folds the overlay in, so the value
                // that comes back is a base with nothing over it, and taking it
                // out of the version is a move rather than a materialisation.
                value: Arc::unwrap_or_clone(next.base),
                map,
                version_label,
                text,
            }
        }))
    }

    /// Why pruning `id`'s covered observations is refused right now, or `None`.
    ///
    /// [`prune_covered_refusal`]'s question asked of a node this state holds,
    /// which is the form the wire and the job want; the Scene tree's menu asks
    /// the free function directly, because the walk that draws it has the node
    /// and not the state.
    pub(crate) fn prune_covered_refusal(&self, id: ReconId) -> Option<String> {
        let busy = self.busy_refusal(id);
        prune_covered_refusal(self.node(id)?, busy.as_deref())
    }

    /// Start a prune of `id`'s covered observations on a worker thread.
    ///
    /// A bulk edit: the retired observations and the points left under the bar
    /// are taken out, so the next version is a whole new base and the surviving
    /// points are renumbered. The caller drops its panel-local caches for the
    /// node afterwards, which the frame that installs the version does off
    /// `Polled::installed`.
    ///
    /// Returns as soon as the worker is running, and **nothing is logged
    /// here**: the entry is the outcome's, written by
    /// [`AppState::poll_background_task`] on the frame the answer lands, from
    /// the instant the operation started and in the name of whoever asked for
    /// it. An `Err` is a refusal to *begin*, logged the way the synchronous
    /// edits log theirs.
    pub fn start_prune_covered_observations(
        &mut self,
        id: ReconId,
        options: &PruneCoveredOptions,
    ) -> Result<(), String> {
        let outcome = match self.prune_covered_observations_job(id, options) {
            Ok(job) => self.start_background_task(Operation::PRUNE_COVERED_OBSERVATIONS, id, job),
            Err(message) => Err(message),
        };
        if let Err(message) = &outcome {
            self.action_log.fail(Kind::Edit, message.clone());
        }
        outcome
    }

    /// The prune itself, as a function of the
    /// [`Progress`](sfmtool_core::progress::Progress) it reports through.
    ///
    /// The closure owns what it reads, the way the retriangulation's does: the
    /// overlay fold, the rule and the write are pure functions of the value at
    /// the cursor, so what crosses to the worker is a clone of
    /// [`EditedReconstruction`] whose `base` is the very `Arc` the node goes on
    /// drawing, and the options.
    pub(crate) fn prune_covered_observations_job(
        &self,
        id: ReconId,
        options: &PruneCoveredOptions,
    ) -> Result<Job, String> {
        let label = self
            .node(id)
            .map(|node| node.label.clone())
            .ok_or_else(|| "That reconstruction is no longer loaded.".to_string())?;
        // The gate is the menu entry's own, so the entry and the edit cannot
        // disagree about when the prune can run.
        if let Some(why) = self.prune_covered_refusal(id) {
            return Err(format!(
                "Prune covered observations in {label} refused: {why}"
            ));
        }
        let edited = self
            .node(id)
            .expect("just resolved")
            .history
            .current()
            .clone();
        let options = *options;
        Ok(Box::new(move |progress| {
            let refuse =
                |why: String| format!("Prune covered observations in {label} refused: {why}");

            // The core operation's own four stages nest directly under the
            // operation's, since this `Progress` is at the top of it, and the
            // overlay fold it does is one of them.
            let (next, map, report) = match prune_covered_observations(&edited, &options, progress)
            {
                Ok(pruned) => pruned,
                // The one error that is not a refusal: the operation was asked
                // to stop and did, which the log words as a cancellation rather
                // than as a failure of the rule.
                Err(PruneCoveredError::Cancelled) => return Finished::Cancelled,
                Err(e) => return Finished::Failed(refuse(e.to_string())),
            };

            let version_label = format!("Pruned covered observations in {label}");
            let text = format!("{version_label}: {}", prune_covered_summary(&report));
            if !report.changed {
                // A prune that retires nothing has run and found nothing to do.
                // A version for it would be a row in the history nobody can
                // tell from one that changed the value.
                return Finished::NoChange(text);
            }
            Finished::Produced {
                // The prune folds the overlay in, so the value that comes back
                // is a base with nothing over it, and taking it out of the
                // version is a move rather than a materialisation.
                value: Arc::unwrap_or_clone(next.base),
                map,
                version_label,
                text,
            }
        }))
    }

    /// The patch radius a gesture that names none takes for `image`, in that
    /// image's pixels: what a cluster seeded at a pixel of it is cut at.
    ///
    /// Data-derived, because the scale a patch wants is the scale the
    /// reconstruction already works at in that view: the median pixel radius of
    /// the patches this image's own observations project to, or the median over
    /// every observation of the node when this image has none, or
    /// [`FALLBACK_PATCH_RADIUS_PX`] when the node has no patch frames at all.
    pub fn default_patch_radius(&self, image: ImageRef) -> f32 {
        let Some(node) = self.node(image.recon) else {
            return FALLBACK_PATCH_RADIUS_PX;
        };
        let edited = node.history.current();
        median_projected_radius(edited, Some(image.index()))
            .or_else(|| median_projected_radius(edited, None))
            .unwrap_or(FALLBACK_PATCH_RADIUS_PX)
    }

    /// Delete one image, and with it its observations and any track that is
    /// left with none.
    ///
    /// A bulk edit: the node's next version is a whole new base, so every
    /// image index at or after the deleted one moves down by one and the
    /// surviving points are renumbered. The caller drops its panel-local caches
    /// for the node afterwards, exactly as it does when a node is closed.
    ///
    /// The entry carries the four stages a bulk edit has -- the overlay fold,
    /// the subset itself, the row map read off its two values, and the version
    /// push -- and is recorded with
    /// [`crate::action_log::ActionLog::record_done`] from the instant below, so
    /// the row says what the edit cost rather than what writing the row cost.
    pub fn delete_image(&mut self, image: ImageRef) -> Result<(), String> {
        let started = Instant::now();
        // The level the Action Log toolbar's checkbox last left, read as the
        // operation starts so that a change to it takes effect on the next one.
        let collector = Collector::new(self.action_log.detailed_timing());
        if let Some(why) = self.busy_refusal(image.recon) {
            return Err(why);
        }
        let index = self
            .scene
            .iter()
            .position(|n| n.id == image.recon)
            .ok_or_else(|| "That reconstruction is no longer loaded.".to_string())?;
        let removed = image.image;
        let node = &self.scene[index];
        let edited = node.history.current();
        if removed as usize >= edited.image_count() {
            return Err("That image is no longer in the reconstruction.".to_string());
        }
        if edited.image_count() == 1 {
            return Err(format!(
                "{} has only one image; deleting it would leave nothing.",
                node.label
            ));
        }
        let name = node.recon().image_table.images[removed as usize]
            .name
            .clone();

        // Materialise only when there is an overlay to fold in; an empty one
        // materialises to its own base, and the subset can read that directly.
        let (materialised, mat_map) =
            if edited.deleted_points.is_empty() && edited.added.points.is_empty() {
                (None, None)
            } else {
                let _phase = collector.phase("materialise");
                let (value, map) = edited.materialize();
                (Some(value), Some(PointMap::Rows(map)))
            };
        let source: &SfmrReconstruction = match materialised.as_ref() {
            Some(value) => value,
            None => &edited.base,
        };

        let keep: Vec<u32> = (0..source.image_count() as u32)
            .filter(|&i| i != removed)
            .collect();
        let subset = {
            let _phase = collector.phase("subset");
            source
                .subset_by_image_indices(&keep, true)
                .map_err(|e| format!("Cannot delete that image: {e}"))?
        };

        // Where each image of `source` went, which is the keep list read the
        // other way round: the deleted one goes nowhere, and everything past it
        // moves down by one.
        let mut image_map: Vec<Option<u32>> = vec![None; source.image_count()];
        for (new, &old) in keep.iter().enumerate() {
            image_map[old as usize] = Some(new as u32);
        }
        // The subset says nothing about which points it dropped, so the map is
        // read off its input and its output.
        let subset_map = {
            let _phase = collector.phase("row map");
            RowMap::by_scan(source, &subset, Some(&image_map))
                .map_err(|e| format!("Cannot delete that image: {e}"))?
        };

        let mut steps = Vec::new();
        steps.extend(mat_map);
        steps.push(PointMap::Rows(subset_map));
        let map = PointMap::Chain(steps);

        let label = node.label.clone();
        let text = format!("Deleted image {name} from {label}");
        // Read before the new base is installed, so the photograph on screen
        // can be found again in it. Deleting the selected image itself leaves
        // no such photograph, and the selection clears.
        let carried = self.selected_image_name(image.recon);
        let node = &mut self.scene[index];
        let serial = {
            let _phase = collector.phase("push version");
            node.history.push(
                EditedReconstruction::new(Arc::new(subset)),
                map,
                text.clone(),
            )
        };
        let parent = version_before(node, serial);
        self.follow_selection_forward(image.recon);
        // Every image index at or past the deleted one moved, so a cached
        // texture named by one of them is now a statement about a different
        // image. The node keeps its identity; what it held about images does
        // not -- and the selection follows the *photograph* rather than the
        // index, which is what an index that moved cannot do for itself.
        self.forget_images_of(image.recon);
        self.follow_image_selection(image.recon, carried.as_deref());
        self.action_log.record_done(
            Kind::Edit,
            started,
            format!("{text} ({parent} → {serial})"),
            collector.take(),
        );
        Ok(())
    }

    /// Re-estimate `image`'s pose against the rest of `source` and install the
    /// answer as `source`'s next version.
    ///
    /// A bulk edit: the resection re-poses one image and re-triangulates the
    /// points it observes, so the next version is a whole new base and the map
    /// is the one `RowMap::by_scan` reads off the call's input and output. The
    /// image table does not move -- a resection re-poses an image, it does not
    /// remove one -- so image indexes, the image and camera selections, and the
    /// decoded pixels keyed by them all still mean what they meant.
    ///
    /// A refused *estimate* pushes no version: installed as the original it
    /// would be a version that moved the points and left the pose alone. See
    /// `specs/gui/edits/resect-image.md`.
    ///
    /// Records its own outcome, success or refusal, as one Action Log entry;
    /// the `Err` is for the caller to know the node's caches are still good, not
    /// to be logged again.
    ///
    /// The entry carries the four stages a bulk edit has -- the overlay fold,
    /// the resection itself, the row map read off its two values, and the
    /// version push -- and is recorded with
    /// [`crate::action_log::ActionLog::record_done`] from the instant below, so
    /// the row says what the resection cost rather than what writing the row
    /// cost.
    pub fn resect_image(
        &mut self,
        source: ReconId,
        image: usize,
        from: ResectFrom,
    ) -> Result<(), String> {
        let started = Instant::now();
        // The level the Action Log toolbar's checkbox last left, read as the
        // operation starts so that a change to it takes effect on the next one.
        let collector = Collector::new(self.action_log.detailed_timing());
        if let Some(why) = self.busy_refusal(source) {
            return Err(why);
        }
        match self.resect_image_inner(source, image, from, &collector) {
            Ok(message) => {
                self.action_log
                    .record_done(Kind::Edit, started, message, collector.take());
                Ok(())
            }
            Err(message) => {
                self.action_log.fail(Kind::Edit, message.clone());
                Err(message)
            }
        }
    }

    /// The edit itself: `Ok` carries the Action Log's sentence, `Err` the
    /// refusal's.
    fn resect_image_inner(
        &mut self,
        source: ReconId,
        image: usize,
        from: ResectFrom,
        collector: &Collector,
    ) -> Result<String, String> {
        let index = self
            .scene
            .iter()
            .position(|n| n.id == source)
            .ok_or_else(|| "That reconstruction is no longer loaded.".to_string())?;
        let label = self.scene[index].label.clone();
        let name = self.scene[index]
            .recon()
            .image_table
            .images
            .get(image)
            .map(|i| i.name.clone())
            .ok_or_else(|| "That image is no longer in the reconstruction.".to_string())?;
        let basename = crate::resect::basename(&name).to_string();
        if from == ResectFrom::Matches {
            self.load_resect_matches(source)
                .map_err(|why| crate::resect::failure_message(&basename, &label, &why))?;
        }

        // Materialise only when there is an overlay to fold in; an empty one
        // materialises to its own base, which the resection can read directly.
        let edited = self.scene[index].history.current();
        let (materialised, mat_map) =
            if edited.deleted_points.is_empty() && edited.added.points.is_empty() {
                (None, None)
            } else {
                let _phase = collector.phase("materialise");
                let (value, map) = edited.materialize();
                (Some(value), Some(PointMap::Rows(map)))
            };
        let source_value: &SfmrReconstruction = match materialised.as_ref() {
            Some(value) => value,
            None => &self.scene[index].history.current().base,
        };

        // One row over the kernel, which takes no `Progress` of its own: the
        // stages inside it are not reachable from here, and this is the row
        // that keeps the resection's own time out of `elsewhere` until they
        // are.
        let outcome = {
            let _phase = collector.phase("resect");
            self.with_resect_source(from, |kind| {
                crate::resect::resect_image_in_place(
                    source_value,
                    image,
                    kind,
                    &crate::resect::ResectImageOptions::default(),
                )
            })
        };
        let (resected, report) = outcome.map_err(|error| {
            crate::resect::failure_message(&basename, &label, &error.to_string())
        })?;

        // The resection may drop a point it could neither re-triangulate nor
        // hold out, and says nothing about which; the map is read off its input
        // and its output. The image table is untouched, so no image map.
        let scan = {
            let _phase = collector.phase("row map");
            RowMap::by_scan(source_value, &resected, None)
                .map_err(|e| crate::resect::failure_message(&basename, &label, &e.to_string()))?
        };
        let mut steps = Vec::new();
        steps.extend(mat_map);
        steps.push(PointMap::Rows(scan));
        let map = PointMap::Chain(steps);

        let text = format!("Resected {basename} ({label})");
        let node = &mut self.scene[index];
        let serial = {
            let _phase = collector.phase("push version");
            node.history.push(
                EditedReconstruction::new(Arc::new(resected)),
                map,
                text.clone(),
            )
        };
        let parent = version_before(node, serial);
        self.follow_selection_forward(source);
        Ok(format!(
            "{text}: {} ({parent} → {serial})",
            crate::resect::outcome_summary(&report)
        ))
    }

    /// Put `image` at `world_from_camera`, and install the answer as its node's
    /// next version.
    ///
    /// The pose is in the **node's own frame**, never the displayed one: the
    /// viewport divides its own pose by the node's `Align to…` transform before
    /// calling, so this method and the offline binding take the same thing (see
    /// `crate::camera_lock::pending_pose`).
    ///
    /// A bulk edit: a pose lives in the base, and the tracks the image observes
    /// are re-triangulated around it, so the next version is a whole new base
    /// under the row map `RowMap::by_scan` reads off the call's input and
    /// output. The move deletes and creates no points, so that scan produces
    /// the identity map -- it is still run rather than assumed, because the map
    /// is a fact about the two values and not about this method.
    ///
    /// The image table does not move, so image indexes and the selections and
    /// decodes keyed by them all still mean what they meant; what the panels
    /// cached *about* the geometry is the caller's to drop, as it is after the
    /// resection and the adjustment. Records its own outcome as one Action Log
    /// entry; the `Err` is for the caller to know the node's caches are still
    /// good, not to be logged again.
    ///
    /// The entry carries the four stages a bulk edit has -- the overlay fold,
    /// the move itself, the row map read off its two values, and the version
    /// push -- and is recorded with
    /// [`crate::action_log::ActionLog::record_done`] from the instant below, so
    /// the row says what the move cost rather than what writing the row cost.
    pub fn move_camera(
        &mut self,
        image: ImageRef,
        world_from_camera: &sfmtool_core::Se3Transform,
    ) -> Result<(), String> {
        let started = Instant::now();
        // The level the Action Log toolbar's checkbox last left, read as the
        // operation starts so that a change to it takes effect on the next one.
        let collector = Collector::new(self.action_log.detailed_timing());
        if let Some(why) = self.busy_refusal(image.recon) {
            return Err(why);
        }
        match self.move_camera_inner(image, world_from_camera, &collector) {
            Ok(message) => {
                self.action_log
                    .record_done(Kind::Edit, started, message, collector.take());
                Ok(())
            }
            Err(message) => {
                self.action_log.fail(Kind::Edit, message.clone());
                Err(message)
            }
        }
    }

    /// The edit itself: `Ok` carries the Action Log's sentence, `Err` the
    /// refusal's.
    fn move_camera_inner(
        &mut self,
        image: ImageRef,
        world_from_camera: &sfmtool_core::Se3Transform,
        collector: &Collector,
    ) -> Result<String, String> {
        let index = self
            .scene
            .iter()
            .position(|n| n.id == image.recon)
            .ok_or_else(|| "That reconstruction is no longer loaded.".to_string())?;
        let label = self.scene[index].label.clone();
        let name = self.scene[index]
            .recon()
            .image_table
            .images
            .get(image.index())
            .map(|i| i.name.clone())
            .ok_or_else(|| "That image is no longer in the reconstruction.".to_string())?;
        let basename = crate::resect::basename(&name).to_string();
        let refuse = |why: String| format!("Cannot move {basename} ({label}): {why}");

        // Materialise only when there is an overlay to fold in; an empty one
        // materialises to its own base, which the move can read directly.
        let edited = self.scene[index].history.current();
        let (materialised, mat_map) =
            if edited.deleted_points.is_empty() && edited.added.points.is_empty() {
                (None, None)
            } else {
                let _phase = collector.phase("materialise");
                let (value, map) = edited.materialize();
                (Some(value), Some(PointMap::Rows(map)))
            };
        let source: &SfmrReconstruction = match materialised.as_ref() {
            Some(value) => value,
            None => &self.scene[index].history.current().base,
        };

        // One row over the kernel, which takes no `Progress` of its own: the
        // re-triangulation inside it is where a slow move's time goes, and this
        // is what keeps that time out of `elsewhere`.
        let (moved, report) = {
            let _phase = collector.phase("move camera");
            sfmtool_core::move_camera(source, image.index(), world_from_camera)
                .map_err(|e| refuse(e.to_string()))?
        };
        // The move deletes and creates no points, so this scan is the identity
        // map -- read off the two values rather than asserted. The image table
        // is untouched, so no image map.
        let scan = {
            let _phase = collector.phase("row map");
            RowMap::by_scan(source, &moved, None).map_err(|e| refuse(e.to_string()))?
        };
        let mut steps = Vec::new();
        steps.extend(mat_map);
        steps.push(PointMap::Rows(scan));
        let map = PointMap::Chain(steps);

        let mut text = format!(
            "Moved camera {basename} ({label}): {:.2} deg, {}",
            report.rotation_deg,
            match report.translation_scene {
                Some(scene) => format!("{scene:.3} scene units"),
                None => format!("{:.4}", report.translation),
            }
        );
        if report.retriangulated > 0 {
            text.push_str(&format!(", {} points re-solved", report.retriangulated));
        }
        let node = &mut self.scene[index];
        let serial = {
            let _phase = collector.phase("push version");
            node.history.push(
                EditedReconstruction::new(Arc::new(moved)),
                map,
                text.clone(),
            )
        };
        let parent = version_before(node, serial);
        self.follow_selection_forward(image.recon);
        let residual = match (report.residual_before_px, report.residual_after_px) {
            (Some(before), Some(after)) => {
                format!(", residual {:.1} → {:.1} px", before[0], after[0])
            }
            _ => String::new(),
        };
        Ok(format!("{text}{residual} ({parent} → {serial})"))
    }

    /// Start a bundle adjustment of `id`'s current value on a worker thread.
    ///
    /// A bulk edit: every posed image's pose, every point's position and, when
    /// the options release it, the shared focal move together, so the next
    /// version is a whole new base under the row map `RowMap::by_scan` reads off
    /// the call's input and output. The map is not decoration here -- a point
    /// the solve leaves unsupported is deleted, and the map is what carries a
    /// selection over that.
    ///
    /// Returns as soon as the worker is running, and **nothing is logged
    /// here**: the entry is the outcome's, written by
    /// [`AppState::poll_background_task`] on the frame the answer lands, from the
    /// instant the operation started and in the name of whoever asked for it.
    /// What this returns an `Err` for is a refusal to *begin*, which is logged
    /// like the refusals the edit used to write itself.
    ///
    /// The image table does not move, so image indexes and the selections keyed
    /// by them still mean what they meant.
    pub fn start_bundle_adjust(
        &mut self,
        id: ReconId,
        options: &sfmtool_core::BundleAdjustOptions,
    ) -> Result<(), String> {
        let outcome = match self.bundle_adjust_job(id, options) {
            Ok(job) => self.start_background_task(Operation::BUNDLE_ADJUST, id, job),
            Err(message) => Err(message),
        };
        if let Err(message) = &outcome {
            self.action_log.fail(Kind::Edit, message.clone());
        }
        outcome
    }

    /// The adjustment itself, as a function of the
    /// [`Progress`](sfmtool_core::progress::Progress) it reports through:
    /// everything between the value at the cursor and the version the GUI
    /// thread will push.
    ///
    /// The closure owns what it reads. The overlay fold, the solve and the row
    /// map are all pure functions of the value at the cursor, so what crosses
    /// to the worker is a clone of [`EditedReconstruction`] -- whose `base` is
    /// the very `Arc` the node goes on drawing, not a copy of it -- and the
    /// options. There is no reference into the scene here, and so nothing for
    /// the GUI thread to be kept out of.
    ///
    /// Refuses before the worker exists, so a refusal is immediate and in the
    /// same words the menu's own gate uses.
    pub(crate) fn bundle_adjust_job(
        &self,
        id: ReconId,
        options: &sfmtool_core::BundleAdjustOptions,
    ) -> Result<Job, String> {
        if let Some(why) = self.busy_refusal(id) {
            return Err(why);
        }
        let node = self
            .node(id)
            .ok_or_else(|| "That reconstruction is no longer loaded.".to_string())?;
        let label = node.label.clone();
        // The gate is the menu entry's own, so the entry and the edit cannot
        // disagree about when the adjustment can run.
        if let Some(why) = crate::bundle_adjust_prompt::refusal(node.history.current()) {
            return Err(format!("Bundle adjust of {label} refused: {why}"));
        }
        let edited = node.history.current().clone();
        let options = options.clone();
        Ok(Box::new(move |progress| {
            let refuse = |why: String| format!("Bundle adjust of {label} refused: {why}");

            // Materialise only when there is an overlay to fold in; an empty
            // one materialises to its own base, which the solve can read
            // directly.
            let (materialised, mat_map) =
                if edited.deleted_points.is_empty() && edited.added.points.is_empty() {
                    (None, None)
                } else {
                    let _phase = progress.phase("materialise");
                    let (value, map) = edited.materialize();
                    (Some(value), Some(PointMap::Rows(map)))
                };
            let source: &SfmrReconstruction = match materialised.as_ref() {
                Some(value) => value,
                None => &edited.base,
            };

            // The kernel's own four stages nest directly under the operation's,
            // since this `Progress` is at the top of it.
            let (adjusted, report) = match sfmtool_core::bundle_adjust(source, &options, progress) {
                Ok(solved) => solved,
                // The one error that is not a refusal: the operation was asked
                // to stop and did, which the log words as a cancellation rather
                // than as a failure of the solve.
                Err(sfmtool_core::reconstruction::bundle_adjust::BundleAdjustError::Cancelled) => {
                    return Finished::Cancelled
                }
                Err(e) => return Finished::Failed(refuse(e.to_string())),
            };
            // The solve drops the points it left unsupported and says how many,
            // not which; the map is read off its input and its output. The image
            // table is untouched, so no image map.
            let scan = {
                let _phase = progress.phase("row map");
                match RowMap::by_scan(source, &adjusted, None) {
                    Ok(scan) => scan,
                    Err(e) => return Finished::Failed(refuse(e.to_string())),
                }
            };
            let mut steps = Vec::new();
            steps.extend(mat_map);
            steps.push(PointMap::Rows(scan));
            let map = PointMap::Chain(steps);

            let mut version_label = format!("Bundle adjusted {label}");
            if report.focal_released {
                version_label.push_str(", focal released");
            }
            let focal = if report.focal_released {
                format!(
                    ", focal {:.1} → {:.1}",
                    report.focal_before, report.focal_after
                )
            } else {
                String::new()
            };
            let deleted = if report.points_deleted > 0 {
                format!(", {} points deleted", report.points_deleted)
            } else {
                String::new()
            };
            let text = format!(
                "{version_label}: {} images, {} points, {} observations, median residual {:.3} → {:.3} px{focal}{deleted}",
                report.images,
                report.points,
                report.observations,
                report.median_residual_before,
                report.median_residual_after,
            );
            Finished::Produced {
                value: adjusted,
                map,
                version_label,
                text,
            }
        }))
    }

    /// Why converting `id` to `embedded_patches` is refused right now, or
    /// `None`.
    ///
    /// [`convert_refusal`]'s question asked of a node this state holds, which
    /// is the form the wire and the job want; the Scene tree's menu asks the
    /// free function directly, because the walk that draws it has the node and
    /// not the state.
    pub(crate) fn convert_to_embedded_patches_refusal(&self, id: ReconId) -> Option<String> {
        let busy = self.busy_refusal(id);
        convert_refusal(self.node(id)?, busy.as_deref())
    }

    /// Start converting `id`'s current value from `sift_files` to
    /// `embedded_patches` on a worker thread.
    ///
    /// A bulk edit, and the minimal conversion: every point keeps its index, its
    /// position and its track, and what changes is how each observation is
    /// located -- a `(u, v)` patch frame per point from the mean viewing
    /// direction, each observation's keypoint copied verbatim from its `.sift`
    /// detection, and each image's identity hash read from the `.sift`
    /// metadata. It is `sfm xform --to-embedded-patches` with that command's
    /// own defaults ([`CONVERSION_EXTENT`]).
    ///
    /// Returns as soon as the worker is running, and **nothing is logged
    /// here**: the entry is the outcome's, written by
    /// [`AppState::poll_background_task`] on the frame the answer lands. An
    /// `Err` is a refusal to *begin*, logged the way the synchronous edits log
    /// theirs.
    pub fn start_convert_to_embedded_patches(&mut self, id: ReconId) -> Result<(), String> {
        let outcome = match self.convert_to_embedded_patches_job(id) {
            Ok(job) => self.start_background_task(Operation::TO_EMBEDDED_PATCHES, id, job),
            Err(message) => Err(message),
        };
        if let Err(message) = &outcome {
            self.action_log.fail(Kind::Edit, message.clone());
        }
        outcome
    }

    /// The conversion itself, as a function of the
    /// [`Progress`](sfmtool_core::progress::Progress) it reports through.
    ///
    /// The closure owns what it reads, the way the adjustment's does: the
    /// overlay fold and the conversion are pure functions of the value at the
    /// cursor, so what crosses to the worker is a clone of
    /// [`EditedReconstruction`] whose `base` is the very `Arc` the node goes on
    /// drawing.
    pub(crate) fn convert_to_embedded_patches_job(&self, id: ReconId) -> Result<Job, String> {
        let label = self
            .node(id)
            .map(|node| node.label.clone())
            .ok_or_else(|| "That reconstruction is no longer loaded.".to_string())?;
        if let Some(why) = self.convert_to_embedded_patches_refusal(id) {
            return Err(format!(
                "Convert to embedded patches of {label} refused: {why}"
            ));
        }
        let edited = self
            .node(id)
            .expect("just resolved")
            .history
            .current()
            .clone();
        Ok(Box::new(move |progress| {
            let refuse =
                |why: String| format!("Convert to embedded patches of {label} refused: {why}");

            // Materialise only when there is an overlay to fold in; an empty
            // one materialises to its own base, which the conversion can read
            // directly.
            let (materialised, mat_map) =
                if edited.deleted_points.is_empty() && edited.added.points.is_empty() {
                    (None, None)
                } else {
                    let _phase = progress.phase("materialise");
                    let (value, map) = edited.materialize();
                    (Some(value), Some(PointMap::Rows(map)))
                };
            let source: &SfmrReconstruction = match materialised.as_ref() {
                Some(value) => value,
                None => &edited.base,
            };

            // The kernel's own three stages nest directly under the
            // operation's, since this `Progress` is at the top of it.
            let converted =
                match source.to_embedded_patches(CONVERSION_NORMAL, CONVERSION_EXTENT, progress) {
                    Ok(converted) => converted,
                    // The one error that is not a refusal: the operation was asked
                    // to stop and did, which the log words as a cancellation.
                    Err(sfmtool_core::reconstruction::ReconstructionError::Cancelled) => {
                        return Finished::Cancelled
                    }
                    Err(e) => return Finished::Failed(refuse(e.to_string())),
                };

            // The identity, stated rather than scanned. The conversion keeps
            // every point at its own index, and `RowMap::by_scan` could not
            // read that off these two values anyway: it identifies a sighting
            // by its feature index, which is exactly the column the conversion
            // replaces, so it would fall back to matching on images alone.
            let mut steps = Vec::new();
            steps.extend(mat_map);
            steps.push(PointMap::Removed(Vec::new()));
            let map = PointMap::Chain(steps);

            let version_label = format!("Converted {label} to embedded patches");
            let text = format!(
                "{version_label}: {} points framed, {} images read",
                converted.point_count(),
                converted.image_count(),
            );
            Finished::Produced {
                value: converted,
                map,
                version_label,
                text,
            }
        }))
    }

    /// Step `id`'s cursor back one version.
    ///
    /// The entry carries the three stages a cursor move has (the step itself,
    /// the selection following it, and the image caches the version it left
    /// behind was holding), and is recorded from the instant below, so the row
    /// says what the move cost rather than what writing the row cost.
    pub fn undo(&mut self, id: ReconId) -> Result<(), String> {
        let started = Instant::now();
        let collector = Collector::new(self.action_log.detailed_timing());
        if let Some(why) = self.busy_refusal(id) {
            return Err(why);
        }
        let Some(index) = self.scene.iter().position(|n| n.id == id) else {
            return Err("That reconstruction is no longer loaded.".to_string());
        };
        // Read before the step: it names the photograph in the version the
        // cursor is leaving, and what it is for is finding that photograph
        // again in the one it lands on.
        let carried = self.selected_image_name(id);
        let step = collector.phase("undo");
        let node = &mut self.scene[index];
        let undone_label = node.history.current_version().label.clone();
        let stepped = {
            let _phase = step.phase("history step");
            node.history.undo()
        };
        let Some((undone, now)) = stepped else {
            return Err(format!("Nothing to undo in {}.", node.label));
        };
        {
            let _phase = step.phase("selection follow");
            self.follow_selection_backward(id, undone);
            self.follow_image_selection(id, carried.as_deref());
        }
        {
            let _phase = step.phase("forget images");
            self.forget_images_of(id);
        }
        drop(step);
        self.action_log.record_done(
            Kind::Edit,
            started,
            format!("Undo: {undone_label} ({undone} → {now})"),
            collector.take(),
        );
        Ok(())
    }

    /// Step `id`'s cursor forward one version.
    ///
    /// The same three stages [`AppState::undo`] reports, under `redo`.
    pub fn redo(&mut self, id: ReconId) -> Result<(), String> {
        let started = Instant::now();
        let collector = Collector::new(self.action_log.detailed_timing());
        if let Some(why) = self.busy_refusal(id) {
            return Err(why);
        }
        let Some(index) = self.scene.iter().position(|n| n.id == id) else {
            return Err("That reconstruction is no longer loaded.".to_string());
        };
        let carried = self.selected_image_name(id);
        let step = collector.phase("redo");
        let node = &mut self.scene[index];
        let stepped = {
            let _phase = step.phase("history step");
            node.history.redo()
        };
        let Some((from, redone)) = stepped else {
            return Err(format!("Nothing to redo in {}.", node.label));
        };
        let redone_label = node.history.current_version().label.clone();
        {
            let _phase = step.phase("selection follow");
            self.follow_selection_forward(id);
            self.follow_image_selection(id, carried.as_deref());
        }
        {
            let _phase = step.phase("forget images");
            self.forget_images_of(id);
        }
        drop(step);
        self.action_log.record_done(
            Kind::Edit,
            started,
            format!("Redo: {redone_label} ({from} → {redone})"),
            collector.take(),
        );
        Ok(())
    }

    /// Move `id`'s cursor straight to the version with serial `serial`.
    ///
    /// The move is one action to the user and one Action Log entry, and it is
    /// the sequence of undos or redos that separates the two versions to
    /// everything that follows a map: the cursor is walked one step at a time
    /// and the selection is put through each step's map in turn, so a jump over
    /// three edits lands the selection exactly where three undos would have.
    /// Refused as a whole when any version it would pass through, the
    /// destination included, has had its value released by the budget: there is
    /// nothing there to show.
    ///
    /// It reports the stages [`AppState::undo`] does, and because the walk is a
    /// loop each of them folds into one row carrying the number of steps.
    pub fn jump_to_version(&mut self, id: ReconId, serial: VersionSerial) -> Result<(), String> {
        let started = Instant::now();
        let collector = Collector::new(self.action_log.detailed_timing());
        if let Some(why) = self.busy_refusal(id) {
            return Err(why);
        }
        let Some(index) = self.scene.iter().position(|n| n.id == id) else {
            return Err("That reconstruction is no longer loaded.".to_string());
        };
        let node = &self.scene[index];
        let Some(target) = node.history.position_of(serial) else {
            return Err(format!("{serial} is not a version of {}.", node.label));
        };
        let cursor = node.history.cursor();
        if target == cursor {
            return Err(format!("{serial} is already what {} shows.", node.label));
        }
        let from = node.history.current_version().serial;
        let span = target.min(cursor)..=target.max(cursor);
        if node.history.versions()[span]
            .iter()
            .any(|v| v.value.is_none())
        {
            return Err(format!(
                "Cannot go to {serial}: its value, or one on the way to it, was released to keep {} inside the history budget.",
                node.label
            ));
        }
        let carried = self.selected_image_name(id);
        let step = collector.phase("go to");
        while self.scene[index].history.cursor() != target {
            let node = &mut self.scene[index];
            let stepping_back = target < node.history.cursor();
            let stepped = {
                let _phase = step.phase("history step");
                node.history.step_towards(target)
            };
            let Some((left, _)) = stepped else {
                break;
            };
            let _phase = step.phase("selection follow");
            if stepping_back {
                self.follow_selection_backward(id, left);
            } else {
                self.follow_selection_forward(id);
            }
            // Per step rather than once at the end, so the stage's row counts
            // the steps like every other stage's. The name is the one the jump
            // started from at each of them, so what decides is the version the
            // walk comes to rest on.
            self.follow_image_selection(id, carried.as_deref());
        }
        let node = &self.scene[index];
        let label = node.history.current_version().label.clone();
        let to = node.history.current_version().serial;
        {
            let _phase = step.phase("forget images");
            self.forget_images_of(id);
        }
        drop(step);
        self.action_log.record_done(
            Kind::Edit,
            started,
            format!("Go to: {label} ({from} → {to})"),
            collector.take(),
        );
        Ok(())
    }

    /// Whether `id` has a version to step back to.
    pub fn can_undo(&self, id: ReconId) -> bool {
        self.node(id).is_some_and(|n| n.history.can_undo())
    }

    /// Whether `id` has a version to step forward to.
    pub fn can_redo(&self, id: ReconId) -> bool {
        self.node(id).is_some_and(|n| n.history.can_redo())
    }

    /// Follow the selected point through the step that produced the version now
    /// at `id`'s cursor: a surviving point keeps its place, a deleted one
    /// clears the selection.
    pub(crate) fn follow_selection_forward(&mut self, id: ReconId) {
        let Some(point) = self.selected_point.filter(|p| p.recon == id) else {
            return;
        };
        let Some(node) = self.node(id) else { return };
        let moved = match node.history.map_into_cursor() {
            Some(map) => map.forward(point.point),
            None => Some(point.point),
        };
        self.selected_point = moved.map(|index| PointRef::new(id, index as usize));
    }

    /// Follow the selected point back through the step `undone` was made by.
    fn follow_selection_backward(&mut self, id: ReconId, undone: crate::document::VersionSerial) {
        let Some(point) = self.selected_point.filter(|p| p.recon == id) else {
            return;
        };
        let Some(node) = self.node(id) else { return };
        let parent = node.history.current_version().serial;
        let moved = match node.history.map_between(parent, undone) {
            Some(map) => map.inverse(point.point),
            None => Some(point.point),
        };
        self.selected_point = moved.map(|index| PointRef::new(id, index as usize));
    }

    /// Where a **background** step's photographs are to come from: what the
    /// cache already holds for the images `needed` names, and the path to
    /// everything else.
    ///
    /// Nothing is decoded and nothing is read here, which is the point: this
    /// runs on the GUI thread and [`ViewSources::decode`] runs on the worker.
    pub(crate) fn view_sources_for(
        &self,
        id: ReconId,
        needed: &[usize],
    ) -> Result<ViewSources, String> {
        let node = self
            .node(id)
            .ok_or_else(|| "That reconstruction is no longer loaded.".to_string())?;
        let recon = node.recon();
        let mut cameras = Vec::with_capacity(recon.image_count());
        let mut poses = Vec::with_capacity(recon.image_count());
        let mut sources = Vec::with_capacity(recon.image_count());
        for (i, im) in recon.image_table.images.iter().enumerate() {
            cameras.push(recon.image_table.cameras[im.camera_index as usize].clone());
            let q = im.quaternion_wxyz;
            poses.push(RigidTransform::from_wxyz_translation(
                [q.w, q.i, q.j, q.k],
                [
                    im.translation_xyz.x,
                    im.translation_xyz.y,
                    im.translation_xyz.z,
                ],
            ));
            sources.push(if !needed.contains(&i) {
                ViewSource::Unused
            } else {
                match self
                    .full_res_cache
                    .get(&ImageRef::new(id, i))
                    .and_then(|slot| slot.as_ref())
                {
                    Some(pyramid) => ViewSource::Decoded(Arc::clone(pyramid)),
                    None => ViewSource::Read(recon.workspace_dir.join(&im.name), im.name.clone()),
                }
            });
        }
        Ok(ViewSources {
            cameras,
            poses,
            sources,
        })
    }

    /// Drop everything this state holds that is keyed by an image of `id`.
    ///
    /// What a bulk edit owes: it renumbers the image table, so a cached decode
    /// would silently become a statement about a different image. The panels'
    /// own texture caches are dropped by the caller, which is where they are
    /// reachable. Hover goes with them -- it is a statement about where a
    /// pointer was over a value that has just been replaced.
    ///
    /// The image **selection** is not dropped here. A cursor move carries it
    /// across with [`AppState::selected_image_name`] and
    /// [`AppState::follow_image_selection`], which is what keeps a run of bench
    /// steps from emptying the Image Detail panel between them.
    fn forget_images_of(&mut self, id: ReconId) {
        self.sift_cache.retain(|image, _| image.recon != id);
        self.full_res_cache.retain(|image, _| image.recon != id);
        self.hovered_image = self.hovered_image.filter(|i| i.recon != id);
        self.hovered_point = self.hovered_point.filter(|p| p.recon != id);
    }

    /// The `.sfmr` name of the image selected in `id`, read **before** a cursor
    /// move.
    ///
    /// A name rather than an index, for the reason the wire addresses images by
    /// one: an index is a coordinate in a particular version's image table, and
    /// a move that crosses a `delete_camera_image` crosses a renumbering of
    /// that table. The name is what the two versions agree on.
    fn selected_image_name(&self, id: ReconId) -> Option<String> {
        let image = self.selected_image.filter(|i| i.recon == id)?;
        let node = self.node(id)?;
        node.recon()
            .image_table
            .images
            .get(image.index())
            .map(|image| image.name.clone())
    }

    /// Put the image selection back on the photograph `name` names in the
    /// version now at `id`'s cursor, and re-derive the lens from it.
    ///
    /// The other half of a cursor move's selection follow: the point half is
    /// [`AppState::follow_selection_forward`], and this is the image half.
    /// Undo, redo and a jump are steps through a node's *history* and not
    /// statements about what the person is looking at, so the photograph on
    /// screen stays on screen -- which for a bench step, an adjustment or a
    /// point edit is every time, none of them touching the image table. Only a
    /// move across a `delete_camera_image` can take the image away, and then
    /// the selection clears because there is nothing left to show.
    ///
    /// Written to the fields rather than through [`AppState::select_image`]:
    /// the photograph did not change, so a "Selected image" row in the Action
    /// Log would be a second event where there was one.
    fn follow_image_selection(&mut self, id: ReconId, name: Option<&str>) {
        let index = name.and_then(|name| {
            let node = self.node(id)?;
            node.recon()
                .image_table
                .images
                .iter()
                .position(|image| image.name == name)
        });
        let Some(index) = index else {
            self.selected_image = self.selected_image.filter(|i| i.recon != id);
            // An intrinsics record has no name to be followed by, so what
            // carries a lens selected on its own is its index still being one
            // the version at the cursor has.
            let cameras = self
                .node(id)
                .map(|node| node.recon().image_table.cameras.len())
                .unwrap_or(0);
            self.selected_camera = self
                .selected_camera
                .filter(|camera| camera.recon != id || camera.index() < cameras);
            return;
        };
        let image = ImageRef::new(id, index);
        self.selected_image = Some(image);
        self.selected_camera = self.camera_of(image);
    }
}

/// The patch radius offered when a node says nothing about what one should be:
/// no patch frames, or none whose projection is readable. Eight pixels is the
/// order of a SIFT keypoint's own support at the scales these captures are
/// detected at, which is the size a hand-placed point is usually after.
pub const FALLBACK_PATCH_RADIUS_PX: f32 = 8.0;

/// How many observations the median is taken over before the walk stops. A
/// median of a few hundred samples is the same number as a median of a million,
/// and the walk runs while a menu is open.
const RADIUS_SAMPLE_CAP: usize = 512;

/// The median pixel radius of the patches `edited`'s observations project to,
/// over one image or over every image, or `None` when none of them projects.
///
/// The radius of one observation is the mean of its projected affine shape's
/// two axis lengths, which is the ellipse the Image Detail panel draws around
/// that keypoint: what the prompt offers is the size of the patches already on
/// screen beside the click.
fn median_projected_radius(edited: &EditedReconstruction, image: Option<usize>) -> Option<f32> {
    let mut radii: Vec<f64> = Vec::new();
    'points: for index in edited.live_indexes() {
        let Some(view) = edited.point(index) else {
            continue;
        };
        for (k, obs) in view.observations().iter().enumerate() {
            let at = obs.image_index as usize;
            if image.is_some_and(|wanted| wanted != at) {
                continue;
            }
            let Some(keypoint) = view.keypoint_xy(k) else {
                continue;
            };
            let Some(shape) = edited.observation_affine_shape(index, at, keypoint) else {
                continue;
            };
            let u = f64::from(shape[0][0]).hypot(f64::from(shape[1][0]));
            let v = f64::from(shape[0][1]).hypot(f64::from(shape[1][1]));
            let radius = 0.5 * (u + v);
            if radius.is_finite() && radius > 0.0 {
                radii.push(radius);
            }
            if radii.len() >= RADIUS_SAMPLE_CAP {
                break 'points;
            }
        }
    }
    if radii.is_empty() {
        return None;
    }
    Some(sfmtool_core::numeric::median_in_place(&mut radii) as f32)
}

/// The serial of the version `serial` was made from, for the log entry.
pub(crate) fn version_before(
    node: &crate::scene::SceneNode,
    serial: crate::document::VersionSerial,
) -> crate::document::VersionSerial {
    let versions = node.history.versions();
    let position = versions
        .iter()
        .position(|v| v.serial == serial)
        .expect("the version was just pushed");
    versions[position.saturating_sub(1)].serial
}
