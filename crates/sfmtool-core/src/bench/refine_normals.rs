// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Refine every finite point's normal, or those of a list of points: for each
//! point, what Track View's *Grown Plane Normal* and *Commit* do to a track
//! put on the bench from it, with every verdict `in` and pinned, written back
//! as one new value. The pairwise-flow normal can be chosen as the step
//! instead.
//!
//! `specs/core/bench/refine-normals.md` is the design. A point's refinement is
//! three bench steps: [`track_of_point`] builds its track, the normal step
//! ([`grown_plane_normal`], or [`pairwise_flow_normal`] where
//! [`RefineNormalsOptions::step`] asks for it) turns its patch about its
//! centre, keeping every keypoint, and renders the bitmap again, and
//! `point_record` builds the record a commit would write. What this module adds
//! is the loop, the skip of points at infinity, the rule for a point the step
//! leaves unchanged or refuses, and the write, which is
//! [`refit_points`](super::refit_points())'s.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use rayon::prelude::*;

use crate::numeric::median_in_place;
use crate::patch::normal_refine::ProjectedImage;
use crate::progress::{Cancelled, Event, Progress};
use crate::reconstruction::data::SfmrReconstruction;
use crate::reconstruction::edited::{EditedReconstruction, PointRecord};
use crate::{progress_info, progress_note};

use super::commit::{point_record, CommitError};
use super::normal::{
    grown_plane_normal, pairwise_flow_normal, GrownPlaneNormalOptions, GrownPlaneOutcome,
    NormalError, NormalEstimate, NormalReport, PairwiseFlowNormalOptions, PairwiseFlowOutcome,
};
use super::refit_points::{
    check_preconditions, stored_in_record_order, write_in_place, RefitPointsError,
};
use super::steps::track_of_point;

/// How a whole-reconstruction normal refinement runs.
#[derive(Debug, Clone, Default)]
pub struct RefineNormalsOptions {
    /// The step each point gets, with its options. The default is the grown
    /// plane normal at `GrownPlaneNormalOptions::default()`, which Track
    /// View's *Grown Plane Normal* runs.
    pub step: RefineNormalsStep,
    /// The points to refine, by index. `None`, the default, refines every
    /// finite point. A point not listed keeps every column it had and is
    /// reported as [`RefineNormalsOutcome::NotAsked`]; a point listed twice is
    /// refined once; an index past the last point refuses the run
    /// ([`RefineNormalsError::PointOutOfRange`]).
    pub point_indexes: Option<Vec<u32>>,
}

/// Which bench normal step each point of a refinement gets, with that step's
/// options.
#[derive(Debug, Clone)]
pub enum RefineNormalsStep {
    /// The grown plane normal ([`grown_plane_normal`]): the one plane the
    /// views agree on, fitted to the photographs view by view. The default.
    GrownPlane(GrownPlaneNormalOptions),
    /// The pairwise-flow normal ([`pairwise_flow_normal`]): the plane on
    /// which the optical flow between pairs of the views agrees, at the
    /// largest support where it agrees. It computes a dense flow per pair
    /// per round, so it costs far more per point than the grown plane.
    PairwiseFlow(PairwiseFlowNormalOptions),
}

impl Default for RefineNormalsStep {
    fn default() -> Self {
        RefineNormalsStep::GrownPlane(GrownPlaneNormalOptions::default())
    }
}

impl RefineNormalsStep {
    /// Whether the step takes its settings, by the step's own check.
    ///
    /// # Errors
    ///
    /// The [`NormalError`] that check returns, naming the setting.
    pub fn check(&self) -> Result<(), NormalError> {
        match self {
            RefineNormalsStep::GrownPlane(options) => options.check(),
            RefineNormalsStep::PairwiseFlow(options) => options.check(),
        }
    }
}

/// What the step that turned a point found, in that step's terms.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NormalTurned {
    /// The grown plane normal turned the patch to its plane.
    GrownPlane {
        /// How many views joined the plane.
        joined: usize,
        /// The plane's weakest `σ`, with each view counted once, is over
        /// [`GrownPlaneNormalOptions::low_confidence_sigma_deg`].
        low_confidence: bool,
        /// How well the views agree at the plane the step found
        /// ([`GrownPlaneSequence::agreement`](super::GrownPlaneSequence::agreement)):
        /// the mean robust grid ZNCC of the views that joined it or had their
        /// shift fitted at it, the reference not counted; `NaN` when none has
        /// a score.
        agreement: f64,
    },
    /// The pairwise-flow normal turned the patch
    /// ([`PairwiseFlowOutcome::Turned`] or [`PairwiseFlowOutcome::SinglePair`]).
    PairwiseFlow {
        /// The point has one pair, read with no agreement to test
        /// ([`PairwiseFlowOutcome::SinglePair`]).
        single_pair: bool,
        /// The radius the answer was read at, in patch half-extents.
        chosen_radius: f64,
        /// The answer's `σ` along its largest axis, degrees.
        sigma_deg: f64,
    },
}

/// Why the step ran on a point and left its normal as it was.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NormalLeftUnchanged {
    /// No view joined the plane, or the reference-view rule found no
    /// reference ([`GrownPlaneOutcome::NoneJoined`]).
    NoneJoined,
    /// A view joined, but the turn is a large one on weak evidence
    /// ([`GrownPlaneOutcome::WeakTurn`]).
    WeakTurn {
        /// The turn the step would have made, degrees.
        turn_deg: f64,
        /// The joined views' mean robust grid ZNCC at the plane found and not
        /// turned to
        /// ([`GrownPlaneSequence::agreement`](super::GrownPlaneSequence::agreement)):
        /// how well they agree on that plane, not on the normal the point
        /// keeps.
        agreement: f64,
    },
    /// The pairwise-flow normal found too little evidence about the tilt
    /// ([`PairwiseFlowOutcome::Undetermined`]).
    Undetermined,
    /// The pairwise-flow normal's pairs are precise but disagree at every
    /// radius, or its answer is too far from the mean viewing direction
    /// ([`PairwiseFlowOutcome::Unstable`]).
    Unstable,
}

/// What a refinement did to one point.
#[derive(Debug, Clone, PartialEq)]
pub enum RefineNormalsOutcome {
    /// The step turned the patch, and its record is the answer.
    Turned {
        /// The turn the patch was given, degrees: the tilt the step reached,
        /// which a view's tilt cap can stop short of the normal it found.
        turn_deg: f64,
        /// Whether the record agrees with the stored one column for column
        /// ([`PointRecord::agrees_with`]) and the stored reference is not a
        /// pick only the display render made, so the point was not rewritten.
        unchanged: bool,
        /// What the step found, in that step's terms.
        by: NormalTurned,
    },
    /// The step ran and left the normal as it was, so
    /// the point keeps every column it had. Where no view joined there is no
    /// plane and no agreement; a large turn on weak evidence carries the
    /// agreement at the plane it found; the pairwise-flow normal's
    /// undetermined and unstable outcomes carry nothing.
    LeftUnchanged(NormalLeftUnchanged),
    /// The step refused the point, and it keeps every column it had.
    NotRefined(NormalError),
    /// The step turned the patch but rendered no bitmap, in a value that
    /// stores one per point, so there is no record to write and the point
    /// keeps every column it had.
    NoBitmap,
    /// The point is at infinity (`w = 0`), whose patch faces along its own
    /// bearing: it was skipped, and keeps every column it had.
    AtInfinity,
    /// The point is not in [`RefineNormalsOptions::point_indexes`]: the run
    /// did not look at it, and it keeps every column it had.
    NotAsked,
}

/// One point's status: its index, which the input and the returned value
/// share, and what the refinement did to it.
#[derive(Debug, Clone, PartialEq)]
pub struct RefinedNormal {
    /// The point's index, in the value the run was handed and in the value it
    /// returned alike.
    pub index: u32,
    /// What happened to it.
    pub outcome: RefineNormalsOutcome,
}

/// What one refinement did, point by point.
///
/// [`Self::points`] is the record, and every count is read off it.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct RefineNormalsReport {
    /// One status per point of the value, in index order.
    pub points: Vec<RefinedNormal>,
}

impl RefineNormalsReport {
    /// The status of the point at `index`, or `None` past the last point.
    pub fn point(&self, index: u32) -> Option<&RefinedNormal> {
        self.points.get(index as usize)
    }

    /// Points the step ran on or tried to: every status but
    /// [`RefineNormalsOutcome::AtInfinity`] and
    /// [`RefineNormalsOutcome::NotAsked`].
    pub fn read(&self) -> usize {
        self.count(|o| {
            !matches!(
                o,
                RefineNormalsOutcome::AtInfinity | RefineNormalsOutcome::NotAsked
            )
        })
    }

    /// Points turned: [`RefineNormalsOutcome::Turned`], unchanged ones
    /// included.
    pub fn turned(&self) -> usize {
        self.count(|o| matches!(o, RefineNormalsOutcome::Turned { .. }))
    }

    /// Of the turned points, the ones whose record agrees with the stored
    /// one, which were not rewritten.
    pub fn unchanged(&self) -> usize {
        self.count(|o| {
            matches!(
                o,
                RefineNormalsOutcome::Turned {
                    unchanged: true,
                    ..
                }
            )
        })
    }

    /// Of the turned points, the ones whose plane the grown plane normal
    /// found with low confidence.
    pub fn low_confidence(&self) -> usize {
        self.count(|o| {
            matches!(
                o,
                RefineNormalsOutcome::Turned {
                    by: NormalTurned::GrownPlane {
                        low_confidence: true,
                        ..
                    },
                    ..
                }
            )
        })
    }

    /// Of the turned points, the ones the pairwise-flow normal read as a
    /// single pair.
    pub fn single_pair(&self) -> usize {
        self.count(|o| {
            matches!(
                o,
                RefineNormalsOutcome::Turned {
                    by: NormalTurned::PairwiseFlow {
                        single_pair: true,
                        ..
                    },
                    ..
                }
            )
        })
    }

    /// Points the step left as they were because no view joined.
    pub fn none_joined(&self) -> usize {
        self.count(|o| {
            matches!(
                o,
                RefineNormalsOutcome::LeftUnchanged(NormalLeftUnchanged::NoneJoined)
            )
        })
    }

    /// Points the step left as they were because the turn was a large one on
    /// weak evidence.
    pub fn weak_turn(&self) -> usize {
        self.count(|o| {
            matches!(
                o,
                RefineNormalsOutcome::LeftUnchanged(NormalLeftUnchanged::WeakTurn { .. })
            )
        })
    }

    /// Points the pairwise-flow normal left as they were because the
    /// photographs carry too little evidence about the tilt.
    pub fn undetermined(&self) -> usize {
        self.count(|o| {
            matches!(
                o,
                RefineNormalsOutcome::LeftUnchanged(NormalLeftUnchanged::Undetermined)
            )
        })
    }

    /// Points the pairwise-flow normal left as they were because its pairs
    /// disagree at every radius or its answer is too far from the mean
    /// viewing direction.
    pub fn unstable(&self) -> usize {
        self.count(|o| {
            matches!(
                o,
                RefineNormalsOutcome::LeftUnchanged(NormalLeftUnchanged::Unstable)
            )
        })
    }

    /// Points kept as they were because the step refused them
    /// ([`RefineNormalsOutcome::NotRefined`]) or rendered them no bitmap
    /// ([`RefineNormalsOutcome::NoBitmap`]).
    pub fn kept(&self) -> usize {
        self.count(|o| {
            matches!(
                o,
                RefineNormalsOutcome::NotRefined(_) | RefineNormalsOutcome::NoBitmap
            )
        })
    }

    /// Points at infinity, which the run skipped.
    pub fn at_infinity(&self) -> usize {
        self.count(|o| matches!(o, RefineNormalsOutcome::AtInfinity))
    }

    /// Points left out of [`RefineNormalsOptions::point_indexes`], which the
    /// run did not look at.
    pub fn not_asked(&self) -> usize {
        self.count(|o| matches!(o, RefineNormalsOutcome::NotAsked))
    }

    /// Points the step refused with an error `is` accepts, such as
    /// `|e| matches!(e, NormalError::TooFewObservations(_))`.
    pub fn with_error(&self, is: impl Fn(&NormalError) -> bool) -> usize {
        self.count(|o| matches!(o, RefineNormalsOutcome::NotRefined(e) if is(e)))
    }

    /// The median turn of the turned points, degrees; `NaN` where none was
    /// turned.
    pub fn median_turn_deg(&self) -> f64 {
        let mut turns: Vec<f64> = self
            .points
            .iter()
            .filter_map(|p| match p.outcome {
                RefineNormalsOutcome::Turned { turn_deg, .. } => Some(turn_deg),
                _ => None,
            })
            .collect();
        if turns.is_empty() {
            f64::NAN
        } else {
            median_in_place(&mut turns)
        }
    }

    /// The mean agreement of the points the grown plane normal turned, a
    /// `NaN` agreement left out; `NaN` where no turned point has one. A point
    /// left unchanged on a large turn on weak evidence does not count: its
    /// agreement is at a plane the run did not write. The pairwise-flow
    /// normal scores no agreement, so a run of it has none.
    pub fn mean_agreement(&self) -> f64 {
        let (sum, n) = self
            .points
            .iter()
            .filter_map(|p| match p.outcome {
                RefineNormalsOutcome::Turned {
                    by: NormalTurned::GrownPlane { agreement, .. },
                    ..
                } if !agreement.is_nan() => Some(agreement),
                _ => None,
            })
            .fold((0.0, 0usize), |(sum, n), z| (sum + z, n + 1));
        if n == 0 {
            f64::NAN
        } else {
            sum / n as f64
        }
    }

    /// How many statuses `keep` accepts.
    fn count(&self, keep: impl Fn(&RefineNormalsOutcome) -> bool) -> usize {
        self.points.iter().filter(|p| keep(&p.outcome)).count()
    }
}

/// Why a refinement produced nothing. Each refusal is read before the copy is
/// taken, so the run writes nothing.
#[derive(Debug, Clone, PartialEq)]
pub enum RefineNormalsError {
    /// The value's observations are `.sift` feature indexes (a `sift_files`
    /// value), and a written keypoint is not a feature, so a refinement, like
    /// a commit ([`CommitError::NotEmbeddedPatches`]), needs an
    /// `embedded_patches` value.
    NoKeypoints,
    /// The points carry no patch frame for the step to turn.
    NoPatchFrames,
    /// No image of the value carries a pose.
    NoPosedImage,
    /// Fewer decoded views were supplied than the value has images.
    ViewsMissing {
        /// How many were supplied.
        got: usize,
        /// How many the value holds.
        expected: usize,
    },
    /// An observation names an image the value does not hold, which only
    /// Rust code that writes `point_set.tracks` directly can hand in
    /// ([`RefitPointsError::ImageOutOfRange`]).
    ImageOutOfRange {
        /// The point whose track names it.
        point: u32,
        /// The index named.
        image: u32,
        /// How many images the value holds.
        image_count: usize,
    },
    /// A setting of [`RefineNormalsOptions::step`] is outside what the step
    /// takes: the step's own check's refusal
    /// ([`NormalError::BadGrownPlaneOption`],
    /// [`NormalError::BadPairwiseFlowOption`] or, for the pairwise-flow
    /// normal's flow settings, [`NormalError::Flow`]).
    BadOption(NormalError),
    /// [`RefineNormalsOptions::point_indexes`] names a point the value does
    /// not hold.
    PointOutOfRange {
        /// The index named.
        point: u32,
        /// How many points the value holds.
        point_count: usize,
    },
    /// The caller asked the run to stop, so nothing was written.
    Cancelled,
}

impl std::fmt::Display for RefineNormalsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RefineNormalsError::NoKeypoints => write!(
                f,
                "refining the normals needs an embedded_patches reconstruction; this one's \
                 observations are .sift features, so convert it to embedded patches first"
            ),
            RefineNormalsError::NoPatchFrames => {
                write!(f, "the points carry no patch frame for the step to turn")
            }
            RefineNormalsError::NoPosedImage => {
                write!(f, "no image of this reconstruction carries a pose")
            }
            RefineNormalsError::ViewsMissing { got, expected } => write!(
                f,
                "{got} decoded views were supplied and the reconstruction has {expected} images"
            ),
            RefineNormalsError::ImageOutOfRange {
                point,
                image,
                image_count,
            } => write!(
                f,
                "point {point}'s track names image {image}, and the reconstruction has \
                 {image_count} images"
            ),
            RefineNormalsError::BadOption(e) => write!(f, "{e}"),
            RefineNormalsError::PointOutOfRange { point, point_count } => write!(
                f,
                "point {point} was asked for, and the reconstruction has {point_count} points"
            ),
            RefineNormalsError::Cancelled => write!(
                f,
                "the refinement was asked to stop before it had an answer, so nothing was \
                 written"
            ),
        }
    }
}

impl std::error::Error for RefineNormalsError {}

impl From<Cancelled> for RefineNormalsError {
    fn from(_: Cancelled) -> Self {
        RefineNormalsError::Cancelled
    }
}

/// The whole-run refusals both runs share, from the refit's check, in this
/// run's own type.
impl From<RefitPointsError> for RefineNormalsError {
    fn from(e: RefitPointsError) -> Self {
        match e {
            RefitPointsError::NoKeypoints => RefineNormalsError::NoKeypoints,
            RefitPointsError::NoPatchFrames => RefineNormalsError::NoPatchFrames,
            RefitPointsError::NoPosedImage => RefineNormalsError::NoPosedImage,
            RefitPointsError::ViewsMissing { got, expected } => {
                RefineNormalsError::ViewsMissing { got, expected }
            }
            RefitPointsError::ImageOutOfRange {
                point,
                image,
                image_count,
            } => RefineNormalsError::ImageOutOfRange {
                point,
                image,
                image_count,
            },
            RefitPointsError::Cancelled => RefineNormalsError::Cancelled,
            // The refit's check reads no noise level.
            RefitPointsError::NoNoiseLevel(why) => {
                unreachable!("the precondition check measures no noise level: {why}")
            }
        }
    }
}

/// Turn every finite point of `recon`, or every finite point of
/// [`RefineNormalsOptions::point_indexes`], to its grown-plane normal, or to
/// its pairwise-flow normal where [`RefineNormalsOptions::step`] asks for it,
/// and hand back a copy of `recon` that holds the answers.
///
/// For each point this runs the steps the bench runs when a person puts the
/// point on the bench, presses *Grown Plane Normal* and presses *Commit*, with
/// every observation `in` and pinned: [`track_of_point`], the step of
/// [`RefineNormalsOptions::step`] ([`grown_plane_normal`] by default, which
/// turns the patch to the one plane its views agree on, renders the bitmap
/// again and reads the track back, or [`pairwise_flow_normal`]), and the
/// record a commit would build from the turned track. The point's centre
/// does not move. No index moves: the returned value has the input's points
/// at the input's indexes, each track keeps its images and its length, and
/// [`RefineNormalsReport::points`] holds one status per point.
///
/// `images` is one [`ProjectedImage`] per image of `recon`, indexed by image
/// index, as [`grown_plane_normal`] takes them. Reading the photographs is
/// the caller's.
///
/// **No keypoint moves.** The step turns each patch about its centre and
/// keeps every keypoint it was given, so the run changes a point's patch
/// frame and what follows from it (its bitmap, the bitmap's reference
/// observation and colour, its observations' readings and blur-matched
/// scores), and a second run starts from the same keypoints as the first. A
/// caller who wants the keypoints
/// relocalized against the turned patches runs
/// [`refit_points`](super::refit_points()) afterwards.
///
/// **Every finite point asked for is refined, held and ranged ones
/// included**, and keeps its constraint: the step moves no centre, so a held
/// coordinate and a ranged distance still describe the point written.
///
/// **A point not asked for is not looked at**
/// ([`RefineNormalsOutcome::NotAsked`]) and keeps every column it had. A
/// point asked for gets the answer a run over every point gives it, since no
/// point's step reads another point's answer.
///
/// **A point at infinity is skipped** ([`RefineNormalsOutcome::AtInfinity`]):
/// its patch faces along its own bearing, and the step refuses it. Nothing
/// here divides by `w`.
///
/// **A point the step leaves unchanged or refuses keeps every column it had**
/// ([`RefineNormalsOutcome::LeftUnchanged`], [`RefineNormalsOutcome::NotRefined`]),
/// the grown plane normal leaving it where no view joined or the turn is
/// large on weak evidence and the pairwise-flow normal where the normal is
/// undetermined or unstable, and so does a point the step renders no bitmap
/// for in a value that stores one per point
/// ([`RefineNormalsOutcome::NoBitmap`]). A point whose record
/// agrees with the stored one is not rewritten, unless its stored reference
/// is a pick only the display render made.
///
/// **The answer is the same for any thread count.** Every point is refined
/// against the value the run was handed, in parallel over the rayon pool, and
/// the records are written at the end. Each step is called with a
/// [`Progress`] that carries the run's cancel flag and reports nothing.
///
/// `progress` names the stages (`refine normals`, which counts the points,
/// and `write back`) and is how the run is asked to stop. The flag is read in
/// front of each stage and of each point's step, and by the step itself; a
/// cancelled run returns [`RefineNormalsError::Cancelled`] and writes
/// nothing.
///
/// # Example
///
/// ```no_run
/// use sfmtool_core::bench::{refine_normals, RefineNormalsOptions};
/// use sfmtool_core::progress::Progress;
/// # fn run(
/// #     recon: &sfmtool_core::SfmrReconstruction,
/// #     views: &[sfmtool_core::patch::normal_refine::ProjectedImage<'_>],
/// # ) -> Result<(), Box<dyn std::error::Error>> {
/// let (next, report) =
///     refine_normals(recon, views, &RefineNormalsOptions::default(), &Progress::none())?;
/// println!(
///     "{} of {} finite points turned, {} at infinity skipped",
///     report.turned(),
///     report.read(),
///     report.at_infinity(),
/// );
/// # let _ = next;
/// # Ok(())
/// # }
/// ```
///
/// # Errors
///
/// [`RefineNormalsError`] states which precondition of the whole run did not
/// hold, that a point asked for is not in it, or that the run was cancelled.
pub fn refine_normals(
    recon: &SfmrReconstruction,
    images: &[ProjectedImage<'_>],
    options: &RefineNormalsOptions,
    progress: &Progress<'_>,
) -> Result<(SfmrReconstruction, RefineNormalsReport), RefineNormalsError> {
    progress.check_cancel()?;
    check_preconditions(recon, images)?;
    options
        .step
        .check()
        .map_err(RefineNormalsError::BadOption)?;
    let point_count = recon.point_count();
    let asked = asked_points(options.point_indexes.as_deref(), point_count)?;
    let [p_refine, p_write] = progress.split([0.95, 0.05]);

    // The copy the write needs, wrapped once for every point's steps.
    let edited = EditedReconstruction::new(Arc::new(recon.clone_for_edit()));
    let total = asked
        .as_ref()
        .map_or(point_count, |a| a.iter().filter(|&&x| x).count());
    // Each point's step reads the run's cancel flag, and what it reports is
    // dropped: per-point phases from a parallel loop are not reported.
    let silent = |_: Event<'_>| {};
    let step_progress = p_refine.tapped(&silent).detailed(false);
    let refined: Option<Vec<(RefinedNormal, Option<PointRecord>)>> = {
        let mut phase = p_refine.phase("refine normals");
        let step = (total / 100).max(1);
        let done = AtomicUsize::new(0);
        let refined = (0..point_count)
            .into_par_iter()
            .map(|p| {
                if asked.as_ref().is_some_and(|a| !a[p]) {
                    let status = RefinedNormal {
                        index: p as u32,
                        outcome: RefineNormalsOutcome::NotAsked,
                    };
                    return Some((status, None));
                }
                if phase.is_cancelled() {
                    return None;
                }
                let answer = refine_one(&edited, images, p as u32, &options.step, &step_progress)?;
                let n = done.fetch_add(1, Ordering::Relaxed) + 1;
                if n.is_multiple_of(step) || n == total {
                    phase.count(n as u64, Some(total as u64), "points");
                }
                Some(answer)
            })
            .collect();
        progress_note!(phase, "{total} points");
        refined
    };
    let refined = refined.ok_or(RefineNormalsError::Cancelled)?;
    progress.check_cancel()?;

    let base = Arc::clone(&edited.base);
    drop(edited);
    let mut out = Arc::unwrap_or_clone(base);
    let mut points = Vec::with_capacity(refined.len());
    {
        let _phase = p_write.phase("write back");
        let mut writes = Vec::new();
        for (status, record) in refined {
            if let Some(record) = record {
                writes.push((status.index as usize, record));
            }
            points.push(status);
        }
        write_in_place(&mut out, writes);
    }
    let report = RefineNormalsReport { points };
    progress_info!(
        progress,
        "{} of {} finite points turned, {} unchanged, {} left as they were, {} at infinity",
        report.turned(),
        report.read(),
        report.none_joined() + report.weak_turn() + report.undetermined() + report.unstable(),
        report.kept(),
        report.at_infinity()
    );
    Ok((out, report))
}

/// Which points the run refines, one flag per point; `None` for every point.
/// A point listed twice is flagged once.
fn asked_points(
    point_indexes: Option<&[u32]>,
    point_count: usize,
) -> Result<Option<Vec<bool>>, RefineNormalsError> {
    let Some(indexes) = point_indexes else {
        return Ok(None);
    };
    let mut asked = vec![false; point_count];
    for &point in indexes {
        let flag = asked
            .get_mut(point as usize)
            .ok_or(RefineNormalsError::PointOutOfRange { point, point_count })?;
        *flag = true;
    }
    Ok(Some(asked))
}

/// One point's refinement: its status, and the record to write where it
/// changed; `None` where the run was cancelled during the point's step.
fn refine_one(
    edited: &EditedReconstruction,
    images: &[ProjectedImage<'_>],
    index: u32,
    step: &RefineNormalsStep,
    progress: &Progress<'_>,
) -> Option<(RefinedNormal, Option<PointRecord>)> {
    let status = |outcome| RefinedNormal { index, outcome };
    let left = |why| Some((status(RefineNormalsOutcome::LeftUnchanged(why)), None));
    let stored = stored_in_record_order(
        edited
            .point(index)
            .expect("every row of a value with no overlay is a live point")
            .to_record(),
    );
    if stored.point.is_at_infinity() {
        return Some((status(RefineNormalsOutcome::AtInfinity), None));
    }
    let track =
        track_of_point(edited, index, None).expect("every row of the value is a live point");
    let answer: Result<(_, NormalReport), NormalError> = match step {
        RefineNormalsStep::GrownPlane(options) => {
            grown_plane_normal(&track, edited, images, options, progress)
        }
        RefineNormalsStep::PairwiseFlow(options) => {
            pairwise_flow_normal(&track, edited, images, options, progress)
        }
    };
    let (turned, report) = match answer {
        Ok(answer) => answer,
        Err(NormalError::Cancelled) => return None,
        Err(e) => return Some((status(RefineNormalsOutcome::NotRefined(e)), None)),
    };
    let by = match &report.estimate {
        NormalEstimate::GrownPlane(sequence) => match sequence.outcome {
            GrownPlaneOutcome::NoneJoined => return left(NormalLeftUnchanged::NoneJoined),
            GrownPlaneOutcome::WeakTurn => {
                return left(NormalLeftUnchanged::WeakTurn {
                    turn_deg: sequence.turn_deg.unwrap_or(f64::NAN),
                    agreement: sequence.agreement(),
                })
            }
            GrownPlaneOutcome::Turned => NormalTurned::GrownPlane {
                joined: sequence.joined(),
                low_confidence: sequence.precision.is_some_and(|p| p.low_confidence),
                agreement: sequence.agreement(),
            },
        },
        NormalEstimate::PairwiseFlow(rounds) => match rounds.outcome {
            PairwiseFlowOutcome::Undetermined => return left(NormalLeftUnchanged::Undetermined),
            PairwiseFlowOutcome::Unstable => return left(NormalLeftUnchanged::Unstable),
            PairwiseFlowOutcome::Turned | PairwiseFlowOutcome::SinglePair => {
                NormalTurned::PairwiseFlow {
                    single_pair: rounds.outcome == PairwiseFlowOutcome::SinglePair,
                    chosen_radius: rounds.chosen_radius.unwrap_or(f64::NAN),
                    sigma_deg: rounds.sigma_deg.unwrap_or(f64::NAN),
                }
            }
        },
        other => unreachable!("each step reports its own estimate, not {other:?}"),
    };

    let mut record = match point_record(edited, &turned) {
        Ok(record) => record,
        Err(CommitError::NoBitmap) => return Some((status(RefineNormalsOutcome::NoBitmap), None)),
        // The run refused a `sift_files` value, and a turned track is at the
        // track stage with its stored position, its patch frame, two or more
        // `in` rows (the step refuses fewer) and a keypoint on every one of
        // them, each in an image of the value; the record is built, never
        // applied, so no edit refuses it.
        Err(e) => unreachable!("a turned track's record is refused only for its bitmap: {e}"),
    };
    // The commit writes every point free; the centre has not moved, so the
    // point's own constraint still describes it.
    record.constraint = stored.constraint;
    let unchanged = stored.agrees_with(&record) && !stored.display_only_reference;
    Some((
        status(RefineNormalsOutcome::Turned {
            turn_deg: report.tilt.degrees,
            unchanged,
            by,
        }),
        (!unchanged).then_some(record),
    ))
}
