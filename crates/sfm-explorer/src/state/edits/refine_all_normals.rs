// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Refine All Normals: every finite point of a node turned to its grown-plane
//! normal, as Track View's *Grown Plane Normal* and *Commit* do for one track
//! with every observation kept, as one version.
//!
//! See `specs/gui/edits/refine-all-normals.md`. The mechanism is
//! [`refine_normals`](sfmtool_core::bench::refine_normals()); what the viewer
//! adds is the gate the menu entry, the wire and the job share
//! ([`refine_refusal`]), the photographs read through the photograph cache on
//! the worker, the fold of the overlay in front of the run, and the Action
//! Log's sentence. It is built the way Refit All Points is, and shares that
//! entry's gate order, its choice of photographs and its counting words
//! ([`super::refit_all_points`]).

use sfmtool_core::bench::{
    refine_normals, NormalError, RefineNormalsError, RefineNormalsOptions, RefineNormalsOutcome,
    RefineNormalsReport,
};
use sfmtool_core::SfmrReconstruction;

use super::refit_all_points::{counted, observed_images, whole_value_bench_refusal};
use crate::action_log::Kind;
use crate::background::{Finished, Job, Operation};
use crate::document::PointMap;
use crate::scene::ReconId;
use crate::state::{AppState, NOT_LOADED};

#[cfg(test)]
mod tests;

/// Why the normals of every point of `node` cannot be refined, or `None`
/// when they can.
///
/// The reasons and their order are Refit All Points'
/// ([`super::refit_refusal`]), asked through the gate the two share: an
/// operation already running on the node (`busy`, with an open camera model
/// proposal's sentence folded in by the Scene tree's menu), observations that
/// carry no inline keypoint, points that carry no patch frame, and no posed
/// image. They are the core run's own refusals ([`RefineNormalsError`]) asked
/// before the gesture. Whether every photograph can be read is not asked
/// here, because it reads files and the menu asks this every frame it is
/// open; the job refuses it on the worker.
pub(crate) fn refine_refusal(node: &crate::scene::SceneNode, busy: Option<&str>) -> Option<String> {
    whole_value_bench_refusal(
        node,
        busy,
        "This reconstruction's observations are .sift feature indexes with no inline \
         keypoints, and refining a normal writes each point's record, keypoints included, \
         as a commit does. \
         Convert to Embedded Patches first.",
        "This reconstruction's points carry no patch frame for the grown-plane normal to \
         turn.",
    )
}

/// How many groups [`left_as_they_were`] names.
const LEFT_REASONS: usize = 9;

/// Where a point the step refused goes in the sentence, and the words that
/// say why: the step's refusal, or `None` for a point the step turned and
/// rendered no bitmap for.
///
/// The match names every [`NormalError`], so a refusal added to the normal
/// steps does not compile until the sentence has words for it.
fn left_as_they_were(error: Option<&NormalError>) -> (usize, &'static str) {
    match error {
        Some(NormalError::TooFewObservations(_)) => (0, "with too few observations"),
        Some(NormalError::NoPatchFrame) => (1, "with no patch frame"),
        None => (2, "with no bitmap to render"),
        Some(NormalError::Tilt(_)) => (3, "whose turn was refused"),
        Some(NormalError::Evaluate(_)) => (4, "whose reading failed"),
        // The run skips a point at infinity before the step, and every point
        // of a value comes onto its track at the track stage.
        Some(NormalError::AtInfinity) => (5, "at infinity"),
        Some(NormalError::ClusterStage) => (6, "at the cluster stage"),
        // The run checks its options once, up front, and runs no other
        // normal step; a cancel ends the run.
        Some(
            NormalError::BadGrownPlaneOption(_)
            | NormalError::BadPieces(_)
            | NormalError::BadOverlap(_)
            | NormalError::TooFewPieces { .. }
            | NormalError::TooFewGridPieces { .. },
        ) => (7, "refused by the step's settings"),
        Some(NormalError::Cancelled) => (8, "whose step was cancelled"),
    }
}

/// What one refinement did, as the Action Log says it after the version
/// label.
///
/// Each count is read off the report's statuses, and a count of zero is left
/// out, in the shape Refit All Points' sentence takes.
fn refine_summary(report: &RefineNormalsReport) -> String {
    let mut text = format!(
        "{} of {} finite points turned",
        report.turned(),
        report.read()
    );
    let unchanged = report.unchanged();
    if unchanged > 0 {
        text.push_str(&format!(", {unchanged} of them as stored"));
    }
    let low = report.low_confidence();
    if low > 0 {
        text.push_str(&format!(", {low} of them with low confidence"));
    }
    let (none_joined, weak_turn) = (report.none_joined(), report.weak_turn());
    if none_joined + weak_turn > 0 {
        let mut why = Vec::new();
        if none_joined > 0 {
            why.push(format!("{none_joined} with no view joined"));
        }
        if weak_turn > 0 {
            why.push(format!("{weak_turn} with a large turn on weak evidence"));
        }
        text.push_str(&format!(
            ", {} normal unchanged ({})",
            counted(
                none_joined + weak_turn,
                "point left its",
                "points left their"
            ),
            why.join(", ")
        ));
    }
    let kept = report.kept();
    if kept > 0 {
        let mut groups = [0usize; LEFT_REASONS];
        let mut words = [""; LEFT_REASONS];
        for point in &report.points {
            let reason = match &point.outcome {
                RefineNormalsOutcome::NotRefined(error) => Some(error),
                RefineNormalsOutcome::NoBitmap => None,
                RefineNormalsOutcome::Turned { .. }
                | RefineNormalsOutcome::LeftUnchanged(_)
                | RefineNormalsOutcome::AtInfinity
                | RefineNormalsOutcome::NotAsked => continue,
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
        text.push_str(&format!(
            ", {} as {} ({})",
            counted(kept, "point left", "points left"),
            if kept == 1 { "it was" } else { "they were" },
            why.join(", ")
        ));
    }
    let at_infinity = report.at_infinity();
    if at_infinity > 0 {
        text.push_str(&format!(
            ", {} skipped",
            counted(at_infinity, "point at infinity", "points at infinity")
        ));
    }
    let median = report.median_turn_deg();
    if median.is_finite() {
        text.push_str(&format!(", median turn {median:.1}\u{b0}"));
    }
    text
}

impl AppState {
    /// Why refining the normals of every point of `id` is refused right now,
    /// or `None`.
    ///
    /// [`refine_refusal`]'s question asked of a node this state holds, which
    /// is the form the wire and the job want; the Scene tree's menu asks the
    /// free function directly.
    pub(crate) fn refine_refusal(&self, id: ReconId) -> Option<String> {
        let busy = self.busy_refusal(id);
        refine_refusal(self.node(id)?, busy.as_deref())
    }

    /// Start a refinement of the normal of every finite point of `id` on a
    /// worker thread.
    ///
    /// A bulk edit: each turned point's patch frame, its patch bitmap and its
    /// readings are written, and no keypoint moves, so the next
    /// version is a whole new base. No point is deleted and none is created,
    /// so the indexes do not move, apart from pending point edits closing up
    /// when the job folds the overlay in.
    ///
    /// Returns as soon as the worker is running, and **nothing is logged
    /// here**: the entry is the outcome's, written by
    /// [`AppState::poll_background_tasks`] on the frame the answer lands. An
    /// `Err` is a refusal to *begin*, logged the way the synchronous edits log
    /// theirs.
    pub(crate) fn start_refine_all_normals(&mut self, id: ReconId) -> Result<Option<u64>, String> {
        let outcome = match self.refine_all_normals_job(id) {
            Ok(job) => self
                .start_background_task(Operation::REFINE_ALL_NORMALS, id, job)
                .map(Some),
            Err(message) => Err(message),
        };
        if let Err(message) = &outcome {
            self.action_log.fail(Kind::Edit, message.clone());
        }
        outcome
    }

    /// The refinement itself, as a function of the
    /// [`Progress`](sfmtool_core::progress::Progress) it reports through.
    ///
    /// What crosses to the worker is a clone of the value at the cursor and
    /// the [`ViewSources`](super::ViewSources) of every image some point
    /// observes, as for Refit All Points. The worker reads the photographs
    /// (`decode images`; one that cannot be read fails the run), folds the
    /// overlay in where it is not empty (`materialise`), and runs the core
    /// refinement, whose stages nest under the operation's.
    pub(crate) fn refine_all_normals_job(&self, id: ReconId) -> Result<Job, String> {
        let label = self
            .node(id)
            .map(|node| node.label.clone())
            .ok_or_else(|| NOT_LOADED.to_string())?;
        // The gate is the menu entry's own, so the entry and the edit cannot
        // disagree about when the refinement can run.
        if let Some(why) = self.refine_refusal(id) {
            return Err(format!("Refine all normals of {label} refused: {why}"));
        }
        let edited = self
            .node(id)
            .expect("just resolved")
            .history
            .current()
            .clone();
        let sources = self.view_sources_for(id, &observed_images(&edited))?;
        Ok(Box::new(move |progress| {
            let refuse = |why: String| format!("Refine all normals of {label} refused: {why}");
            if progress.is_cancelled() {
                return Finished::Cancelled;
            }
            // The decode counts its images to the end of the range it is
            // handed, so it gets a share of the bar and the refinement the
            // rest, as for Refit All Points.
            let [p_decode, p_refine] = progress.split([1.0, 9.0]);
            let decoded = match sources.decode(&p_decode) {
                Ok(decoded) => decoded,
                Err(_) if progress.is_cancelled() => return Finished::Cancelled,
                Err(e) => return Finished::Failed(refuse(e)),
            };
            if progress.is_cancelled() {
                return Finished::Cancelled;
            }

            let (materialised, mat_map) =
                if edited.deleted_points.is_empty() && edited.added.points.is_empty() {
                    (None, None)
                } else {
                    let _phase = p_refine.phase("materialise");
                    let (value, map) = edited.materialize();
                    (Some(value), Some(PointMap::Rows(map)))
                };
            let source: &SfmrReconstruction = match materialised.as_ref() {
                Some(value) => value,
                None => &edited.base,
            };

            let views = decoded.views();
            let (next, report) =
                match refine_normals(source, &views, &RefineNormalsOptions::default(), &p_refine) {
                    Ok(refined) => refined,
                    Err(RefineNormalsError::Cancelled) => return Finished::Cancelled,
                    Err(e) => return Finished::Failed(refuse(e.to_string())),
                };

            // The refinement moves no index: the identity, stated, after the
            // fold.
            let mut steps = Vec::new();
            steps.extend(mat_map);
            steps.push(PointMap::Removed(Vec::new()));
            let version_label = format!("Refined the normals of {label}");
            let text = format!("{version_label}: {}", refine_summary(&report));
            Finished::Produced {
                value: next,
                map: PointMap::Chain(steps),
                version_label,
                text,
            }
        }))
    }
}
