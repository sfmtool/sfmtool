// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Refit All Points: every point of a node fitted to the render of its
//! reference observation, as Track View's *Fit* and *Commit* do for one track
//! with every observation kept, as one version.
//!
//! See `specs/gui/edits/refit-all-points.md`. The mechanism is
//! [`refit_points`](sfmtool_core::bench::refit_points()); what the viewer adds
//! is the gate the menu entry, the wire and the job share ([`refit_refusal`]),
//! the photographs read through the photograph cache on the worker, the fold
//! of the overlay in front of the run, and the Action Log's sentence.

use sfmtool_core::bench::{
    refit_points, FitError, RefitOutcome, RefitPointsError, RefitPointsOptions, RefitPointsReport,
};
use sfmtool_core::{EditedReconstruction, SfmrReconstruction};

use crate::action_log::Kind;
use crate::background::{Finished, Job, Operation};
use crate::document::PointMap;
use crate::scene::ReconId;
use crate::state::{AppState, NOT_LOADED};

#[cfg(test)]
mod tests;

/// Why every point of `node` cannot be refit, or `None` when it can.
///
/// The reasons a caller can see without reading a photograph, asked in this
/// order: an operation already running on the node (`busy`, which the Scene
/// tree's menu passes with an open camera model proposal's sentence folded in),
/// observations that carry no inline keypoint, points that carry no patch
/// frame, and no posed image. They are the core run's own refusals
/// ([`RefitPointsError`]) asked before the gesture. Whether every photograph
/// can be read and whether the reconstruction's noise level can be measured
/// are not asked here: the first reads files and the second walks every
/// observation, and the menu asks this every frame it is open. The job refuses
/// both on the worker.
///
/// A free function over the node and the busy sentence for the reason
/// [`super::retriangulate_refusal`] is one: the menu is drawn inside a walk that
/// has given up its borrow of the state.
pub(crate) fn refit_refusal(node: &crate::scene::SceneNode, busy: Option<&str>) -> Option<String> {
    if let Some(why) = busy {
        return Some(why.to_string());
    }
    let edited = node.history.current();
    if !edited.has_keypoints() {
        return Some(
            "This reconstruction's observations are .sift feature indexes with no inline \
             keypoints, and a refit rewrites a keypoint per observation. Convert to Embedded \
             Patches first."
                .to_string(),
        );
    }
    if !edited.has_patch_frames() {
        return Some(
            "This reconstruction's points carry no patch frame, and a fit registers a point's \
             frame into every image that sees it."
                .to_string(),
        );
    }
    (edited.posed_lens_count() == 0)
        .then(|| "No image of this reconstruction carries a pose.".to_string())
}

/// How many groups [`left_as_they_were`] names.
const LEFT_REASONS: usize = 9;

/// Where a point the refit left as it was goes in the sentence, and the words
/// that say why: the fit's refusal, or `None` for a point the fit placed and
/// rendered no bitmap for.
///
/// The match names every [`FitError`], so a refusal added to the fit does not
/// compile until the sentence has words for it.
fn left_as_they_were(error: Option<&FitError>) -> (usize, &'static str) {
    match error {
        Some(FitError::TooFewObservations(_)) => (0, "with too few observations"),
        Some(FitError::TooFewRays { .. }) => (1, "with too few rays"),
        Some(FitError::Triangulation) => (2, "whose rays do not triangulate"),
        Some(FitError::NoPatchFrame) => (3, "with no patch frame"),
        None => (4, "with no bitmap to render"),
        Some(FitError::TooLarge { .. }) | Some(FitError::OutOfMemory { .. }) => {
            (5, "too large to read in memory")
        }
        Some(FitError::ViewsMissing { .. }) | Some(FitError::NoView { .. }) => {
            (6, "with an image that has no view")
        }
        // Neither reaches a point in a run that ends with a version: the run
        // measures its noise level once, up front, and a cancel ends it.
        Some(FitError::NoNoiseLevel(_)) => (7, "with no noise level"),
        Some(FitError::Cancelled) => (8, "whose fit was cancelled"),
    }
}

/// `n` and the noun, plural unless `n` is one.
fn counted(n: usize, one: &str, many: &str) -> String {
    format!("{n} {}", if n == 1 { one } else { many })
}

/// What one refit did, as the Action Log says it after the version label.
///
/// Each count is read off the report's statuses, and a count of zero is left
/// out, in the shape the retriangulation's sentence takes.
fn refit_summary(report: &RefitPointsReport) -> String {
    let mut text = format!("{} of {} points refit", report.refit(), report.read());
    let unchanged = report.unchanged();
    if unchanged > 0 {
        text.push_str(&format!(", {unchanged} of them as stored"));
    }
    let ranged = report.ranged();
    if ranged > 0 {
        text.push_str(&format!(", {ranged} of them ranged at their distance"));
    }
    let released = report.released();
    if released == 1 {
        text.push_str(
            ", 1 ranged point released: its reference image has no pose and it became a \
             direction",
        );
    } else if released > 1 {
        text.push_str(&format!(
            ", {released} ranged points released: their reference images have no pose and \
             they became directions"
        ));
    }
    let crossed = report.crossed();
    if crossed > 0 {
        text.push_str(&format!(", {crossed} crossed to or from infinity"));
    }
    let kept_at_seed = report.kept_at_seed();
    if kept_at_seed > 0 {
        text.push_str(&format!(
            ", {} past the shift bound",
            counted(
                kept_at_seed,
                "observation kept its keypoint",
                "observations kept their keypoint"
            )
        ));
    }
    let kept = report.kept();
    if kept > 0 {
        let mut groups = [0usize; LEFT_REASONS];
        let mut words = [""; LEFT_REASONS];
        for point in &report.points {
            let reason = match &point.outcome {
                RefitOutcome::NotFitted(error) => Some(error),
                RefitOutcome::NoBitmap => None,
                RefitOutcome::Refit { .. } | RefitOutcome::Held => continue,
            };
            let (k, w) = left_as_they_were(reason);
            groups[k] += 1;
            words[k] = w;
        }
        let why: Vec<String> = groups
            .iter()
            .zip(words)
            .filter(|(n, _)| **n > 0)
            .map(|(n, w)| format!("{n} {w}"))
            .collect();
        text.push_str(&format!(", {kept} left as they were ({})", why.join(", ")));
    }
    let held = report.held();
    if held > 0 {
        text.push_str(&format!(", {held} held"));
    }
    let median_shift = report.median_shift();
    if median_shift.is_finite() {
        text.push_str(&format!(", median shift {median_shift:.4}"));
    }
    text
}

/// Every image some live point of `edited` observes, ascending: the
/// photographs a refit reads.
fn observed_images(edited: &EditedReconstruction) -> Vec<usize> {
    let mut seen = vec![false; edited.image_count()];
    for index in edited.live_indexes() {
        if let Some(point) = edited.point(index) {
            for observation in point.observations() {
                if let Some(slot) = seen.get_mut(observation.image_index as usize) {
                    *slot = true;
                }
            }
        }
    }
    seen.iter()
        .enumerate()
        .filter(|(_, &seen)| seen)
        .map(|(i, _)| i)
        .collect()
}

impl AppState {
    /// Why refitting every point of `id` is refused right now, or `None`.
    ///
    /// [`refit_refusal`]'s question asked of a node this state holds, which is
    /// the form the wire and the job want; the Scene tree's menu asks the free
    /// function directly.
    pub(crate) fn refit_refusal(&self, id: ReconId) -> Option<String> {
        let busy = self.busy_refusal(id);
        refit_refusal(self.node(id)?, busy.as_deref())
    }

    /// Start a refit of every point of `id` on a worker thread.
    ///
    /// A bulk edit: every point's keypoints, geometry, patch bitmap and
    /// readings are read again from the photographs, so the next version is a
    /// whole new base. No point is deleted and none is created, so the indexes
    /// do not move, apart from pending point edits closing up when the job
    /// folds the overlay in.
    ///
    /// Returns as soon as the worker is running, and **nothing is logged
    /// here**: the entry is the outcome's, written by
    /// [`AppState::poll_background_tasks`] on the frame the answer lands. An
    /// `Err` is a refusal to *begin*, logged the way the synchronous edits log
    /// theirs.
    pub(crate) fn start_refit_all_points(&mut self, id: ReconId) -> Result<Option<u64>, String> {
        let outcome = match self.refit_all_points_job(id) {
            Ok(job) => self
                .start_background_task(Operation::REFIT_ALL_POINTS, id, job)
                .map(Some),
            Err(message) => Err(message),
        };
        if let Err(message) = &outcome {
            self.action_log.fail(Kind::Edit, message.clone());
        }
        outcome
    }

    /// The refit itself, as a function of the
    /// [`Progress`](sfmtool_core::progress::Progress) it reports through.
    ///
    /// What crosses to the worker is a clone of the value at the cursor, whose
    /// `base` is the `Arc` the node goes on drawing, and the
    /// [`ViewSources`](super::ViewSources) of every image some point observes.
    /// The worker reads the photographs (`decode images`; one that cannot be
    /// read fails the run), folds the overlay in where it is not empty
    /// (`materialise`), and runs the core refit, whose stages nest under the
    /// operation's.
    pub(crate) fn refit_all_points_job(&self, id: ReconId) -> Result<Job, String> {
        let label = self
            .node(id)
            .map(|node| node.label.clone())
            .ok_or_else(|| NOT_LOADED.to_string())?;
        // The gate is the menu entry's own, so the entry and the edit cannot
        // disagree about when the refit can run.
        if let Some(why) = self.refit_refusal(id) {
            return Err(format!("Refit all points of {label} refused: {why}"));
        }
        let edited = self
            .node(id)
            .expect("just resolved")
            .history
            .current()
            .clone();
        let sources = self.view_sources_for(id, &observed_images(&edited))?;
        Ok(Box::new(move |progress| {
            let refuse = |why: String| format!("Refit all points of {label} refused: {why}");
            if progress.is_cancelled() {
                return Finished::Cancelled;
            }
            // The decode counts its images to the end of the range it is
            // handed, so it gets a share of the bar and the refit the rest: on
            // the whole range it would fill the bar before the first point is
            // fitted. Reading the photographs takes seconds; fitting every
            // point is most of the run.
            let [p_decode, p_refit] = progress.split([1.0, 9.0]);
            // Every photograph, or the run does not go ahead: a refit of most
            // points with those of one image silently skipped is harder to read
            // than a refusal that names the file.
            let decoded = match sources.decode(&p_decode) {
                Ok(decoded) => decoded,
                Err(_) if progress.is_cancelled() => return Finished::Cancelled,
                Err(e) => return Finished::Failed(refuse(e)),
            };
            if progress.is_cancelled() {
                return Finished::Cancelled;
            }

            // Materialise only when there is an overlay to fold in; an empty
            // one materialises to its own base, which the refit reads directly.
            let (materialised, mat_map) =
                if edited.deleted_points.is_empty() && edited.added.points.is_empty() {
                    (None, None)
                } else {
                    let _phase = p_refit.phase("materialise");
                    let (value, map) = edited.materialize();
                    (Some(value), Some(PointMap::Rows(map)))
                };
            let source: &SfmrReconstruction = match materialised.as_ref() {
                Some(value) => value,
                None => &edited.base,
            };

            let views = decoded.views();
            let (next, report) =
                match refit_points(source, &views, &RefitPointsOptions::default(), &p_refit) {
                    Ok(refit) => refit,
                    // The one error that is not a refusal: the run was asked to
                    // stop and did, which the log words as a cancellation.
                    Err(RefitPointsError::Cancelled) => return Finished::Cancelled,
                    Err(e) => return Finished::Failed(refuse(e.to_string())),
                };

            // The refit moves no index: the identity, stated, after the fold.
            let mut steps = Vec::new();
            steps.extend(mat_map);
            steps.push(PointMap::Removed(Vec::new()));
            let version_label = format!("Refit {label}");
            let text = format!("{version_label}: {}", refit_summary(&report));
            Finished::Produced {
                value: next,
                map: PointMap::Chain(steps),
                version_label,
                text,
            }
        }))
    }
}
