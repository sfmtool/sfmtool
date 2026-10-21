// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Refit every point of a reconstruction: for each point, what Track View's
//! *Fit* and *Commit* do to a track put on the bench from it, with every
//! verdict `in` and pinned, written back as one new value.
//!
//! `specs/core/bench/refit-points.md` is the design. A point's refit is three
//! bench steps: [`track_of_point`] builds its track, `fit_track` aligns the
//! other observations to the render of its reference observation,
//! re-triangulates it and renders its bitmap again, and `point_record` builds
//! the record a commit would write. What this module adds is the loop, the rule
//! for a held or ranged point, the rule for a point the fit refuses, and the
//! write of every record into a copy of the value at the point's own rows.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use ndarray::Axis;
use rayon::prelude::*;

use sfmtool_sfmr_format::{
    NO_REFERENCE_IMAGE, POINT_CONSTRAINT_FREE, POINT_CONSTRAINT_HELD, POINT_CONSTRAINT_RANGED,
};

use crate::numeric::median_in_place;
use crate::patch::normal_refine::ProjectedImage;
use crate::progress::{Cancelled, Progress};
use crate::reconstruction::bundle_adjust::is_posed;
use crate::reconstruction::data::{
    ImageSet, ObservationReading, ObservationReadings, ObservationSource, PointSet,
    SfmrReconstruction,
};
use crate::reconstruction::edited::{reprojection_noise_level, EditedReconstruction, PointRecord};
use crate::reconstruction::triangulation::points::PointDistance;
use crate::reconstruction::triangulation::retriangulate::reference_camera_center;
use crate::{progress_info, progress_note};

use super::classify::TrackClassification;
use super::commit::{point_record, CommitError};
use super::fit::{fit_preconditions, fit_track, stated_or_measured_noise, FitError, FitOptions};
use super::steps::track_of_point;
use super::track::{Stage, BENCH_MAX_SHIFT_PX};

/// How a whole-reconstruction refit runs.
#[derive(Debug, Clone)]
pub struct RefitPointsOptions {
    /// The fit each point gets. Track View's Fit passes
    /// `FitOptions::default()` with the noise level its panel holds in
    /// `sigma_px`. `sigma_px: None` is measured once for the whole run from
    /// the value it was handed, not once per point.
    pub fit: FitOptions,
    /// How far, in grid px, a fit may move a sighting before it keeps its
    /// keypoint instead: the bound
    /// [`Thresholds::max_shift_px`](super::track::Thresholds::max_shift_px)
    /// sets for a track on the bench. [`BENCH_MAX_SHIFT_PX`] (6) by default,
    /// the value every track put on the bench starts with.
    pub max_shift_px: f64,
}

impl Default for RefitPointsOptions {
    fn default() -> Self {
        Self {
            fit: FitOptions::default(),
            max_shift_px: BENCH_MAX_SHIFT_PX,
        }
    }
}

/// How a refit point's constraint was applied to its answer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RefitRange {
    /// The point is free, and its coordinate is the fit's classification.
    Free,
    /// The point is ranged, and its direction was read at its distance by the
    /// triangulation's distance rule. Its constraint is kept.
    Applied,
    /// The point is ranged at a finite distance from an image that has no
    /// pose, so there is no origin to measure from, and it was fitted free.
    Unresolved {
        /// Whether the fit made it a direction, which a finite distance cannot
        /// describe, so its constraint was released and it was written free.
        /// Where it stayed finite, its constraint is kept.
        released: bool,
    },
}

/// What a refit did to one point.
#[derive(Debug, Clone, PartialEq)]
pub enum RefitOutcome {
    /// The point was fitted, and its record is the answer.
    Refit {
        /// How many observations the localizer placed
        /// ([`FitReport::placed`](super::fit::FitReport::placed)).
        placed: usize,
        /// How many sightings the localizer would have moved past
        /// [`RefitPointsOptions::max_shift_px`] and so kept their keypoints
        /// ([`FitReport::kept_at_seed`](super::fit::FitReport::kept_at_seed)).
        kept_at_seed: usize,
        /// Whether the point went from a position to a direction or back.
        crossed: bool,
        /// How far a point that stayed a position moved, in the value's own
        /// units; `None` for a direction, or a point that crossed.
        shift: Option<f64>,
        /// The fit's point-or-bearing classification; `None` for a point
        /// whose coordinate the distance rule decided.
        classification: Option<TrackClassification>,
        /// How the point's constraint was applied.
        range: RefitRange,
        /// Whether the record agrees with the stored one column for column
        /// ([`PointRecord::agrees_with`]) and the stored reference is not a
        /// pick only the display render made, so the point was not rewritten.
        unchanged: bool,
    },
    /// The value holds the point's coordinate, so it was left out of the run.
    Held,
    /// The fit refused the point, and it keeps every column it had.
    NotFitted(FitError),
    /// The fit placed the point but rendered no bitmap, in a value that stores
    /// one per point, so there is no record to write and the point keeps every
    /// column it had.
    NoBitmap,
}

/// One point's status: its index, which the input and the returned value
/// share, and what the refit did to it.
#[derive(Debug, Clone, PartialEq)]
pub struct RefitPoint {
    /// The point's index, in the value the run was handed and in the value it
    /// returned alike.
    pub index: u32,
    /// What happened to it.
    pub outcome: RefitOutcome,
}

/// What one refit did, point by point.
///
/// [`Self::points`] is the record, and every count is read off it.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct RefitPointsReport {
    /// One status per point of the value, in index order.
    pub points: Vec<RefitPoint>,
}

impl RefitPointsReport {
    /// The status of the point at `index`, or `None` past the last point.
    pub fn point(&self, index: u32) -> Option<&RefitPoint> {
        self.points.get(index as usize)
    }

    /// Points the run fitted or tried to: every status but
    /// [`RefitOutcome::Held`].
    pub fn read(&self) -> usize {
        self.count(|o| !matches!(o, RefitOutcome::Held))
    }

    /// Points fitted: [`RefitOutcome::Refit`], unchanged ones included.
    pub fn refit(&self) -> usize {
        self.count(|o| matches!(o, RefitOutcome::Refit { .. }))
    }

    /// Of the refit points, the ones whose record agrees with the stored one,
    /// which were not rewritten.
    pub fn unchanged(&self) -> usize {
        self.count(|o| {
            matches!(
                o,
                RefitOutcome::Refit {
                    unchanged: true,
                    ..
                }
            )
        })
    }

    /// Of the refit points, the ones that went from a position to a direction
    /// or back.
    pub fn crossed(&self) -> usize {
        self.count(|o| matches!(o, RefitOutcome::Refit { crossed: true, .. }))
    }

    /// Points left as they were: the fit refused them
    /// ([`RefitOutcome::NotFitted`]) or rendered them no bitmap
    /// ([`RefitOutcome::NoBitmap`]).
    pub fn kept(&self) -> usize {
        self.count(|o| matches!(o, RefitOutcome::NotFitted(_) | RefitOutcome::NoBitmap))
    }

    /// Points the value holds at their coordinate, which the run left out.
    pub fn held(&self) -> usize {
        self.count(|o| matches!(o, RefitOutcome::Held))
    }

    /// Refit points whose direction was read at their distance
    /// ([`RefitRange::Applied`]).
    pub fn ranged(&self) -> usize {
        self.count(|o| {
            matches!(
                o,
                RefitOutcome::Refit {
                    range: RefitRange::Applied,
                    ..
                }
            )
        })
    }

    /// Ranged points whose constraint was released
    /// ([`RefitRange::Unresolved`] with `released`).
    pub fn released(&self) -> usize {
        self.count(|o| {
            matches!(
                o,
                RefitOutcome::Refit {
                    range: RefitRange::Unresolved { released: true },
                    ..
                }
            )
        })
    }

    /// Sightings that kept their keypoint past the shift bound, summed over
    /// the refit points.
    pub fn kept_at_seed(&self) -> usize {
        self.points
            .iter()
            .map(|p| match p.outcome {
                RefitOutcome::Refit { kept_at_seed, .. } => kept_at_seed,
                _ => 0,
            })
            .sum()
    }

    /// Points the fit refused with an error `is` accepts, such as
    /// `|e| matches!(e, FitError::TooFewRays { .. })`.
    pub fn with_error(&self, is: impl Fn(&FitError) -> bool) -> usize {
        self.count(|o| matches!(o, RefitOutcome::NotFitted(e) if is(e)))
    }

    /// The median distance a refit point that stayed a position moved, in the
    /// value's own units; `NaN` where none did. Over the points that were
    /// positions before and after alone, since a crossing has no distance.
    pub fn median_shift(&self) -> f64 {
        let mut shifts: Vec<f64> = self
            .points
            .iter()
            .filter_map(|p| match p.outcome {
                RefitOutcome::Refit { shift, .. } => shift,
                _ => None,
            })
            .collect();
        if shifts.is_empty() {
            f64::NAN
        } else {
            median_in_place(&mut shifts)
        }
    }

    /// How many statuses `keep` accepts.
    fn count(&self, keep: impl Fn(&RefitOutcome) -> bool) -> usize {
        self.points.iter().filter(|p| keep(&p.outcome)).count()
    }
}

/// Why a refit produced nothing. Each refusal is read before the copy is
/// taken, so the run writes nothing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RefitPointsError {
    /// The value's observations are `.sift` feature indexes (a `sift_files`
    /// value), and a fitted keypoint is not a feature, so a refit, like a
    /// commit ([`CommitError::NotEmbeddedPatches`]), needs an
    /// `embedded_patches` value.
    NoKeypoints,
    /// The points carry no patch frame, and a fit registers the patch frame into
    /// every view ([`FitError::NoPatchFrame`]).
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
    /// An observation names an image the value does not hold. A load,
    /// `from_sfmr_data` and the Python `clone_with_changes` refuse such a value
    /// ([`PointSet::validate_track_image_indexes`](crate::PointSet::validate_track_image_indexes)),
    /// and the editors build their tracks from a value that passed, so only
    /// Rust code that writes `point_set.tracks` directly can hand one in.
    ImageOutOfRange {
        /// The point whose track names it.
        point: u32,
        /// The index named.
        image: u32,
        /// How many images the value holds.
        image_count: usize,
    },
    /// No noise level to weight the rays by: no `sigma_px` was given and the
    /// value's reprojection noise could not be measured, or the one given is
    /// not a finite positive number. The sentence says why.
    NoNoiseLevel(String),
    /// The caller asked the run to stop, so nothing was written.
    Cancelled,
}

impl std::fmt::Display for RefitPointsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RefitPointsError::NoKeypoints => write!(
                f,
                "a refit needs an embedded_patches reconstruction; this one's observations \
                 are .sift features, so convert it to embedded patches first"
            ),
            RefitPointsError::NoPatchFrames => write!(
                f,
                "the points carry no patch frame for the fit to register into each view"
            ),
            RefitPointsError::NoPosedImage => {
                write!(f, "no image of this reconstruction carries a pose")
            }
            RefitPointsError::ViewsMissing { got, expected } => write!(
                f,
                "{got} decoded views were supplied and the reconstruction has {expected} images"
            ),
            RefitPointsError::ImageOutOfRange {
                point,
                image,
                image_count,
            } => write!(
                f,
                "point {point}'s track names image {image}, and the reconstruction has \
                 {image_count} images"
            ),
            RefitPointsError::NoNoiseLevel(why) => write!(
                f,
                "the points cannot be classified as points or bearings: {why}"
            ),
            RefitPointsError::Cancelled => write!(
                f,
                "the refit was asked to stop before it had an answer, so nothing was written"
            ),
        }
    }
}

impl std::error::Error for RefitPointsError {}

impl From<Cancelled> for RefitPointsError {
    fn from(_: Cancelled) -> Self {
        RefitPointsError::Cancelled
    }
}

/// Refit every point of `recon` to the render of its reference observation
/// and hand back a copy of `recon` that holds the answers.
///
/// For each point this runs the steps the bench runs when a person puts the
/// point on the bench, presses *Fit* and presses *Commit*, with every
/// observation `in` and pinned: [`track_of_point`], the fit at the track stage
/// with [`RefitPointsOptions::fit`] and [`RefitPointsOptions::max_shift_px`]
/// (every sighting aligned to the render of the point's reference observation,
/// the point re-triangulated, its patch frame placed at the answer and its bitmap
/// rendered again, then evaluated), and the record a commit would build from
/// the fitted track. No index moves: the returned value has the input's points
/// at the input's indexes, each track keeps its images and its length, and
/// [`RefitPointsReport::points`] holds one status per point.
///
/// `images` is one [`ProjectedImage`] per image of `recon`, indexed by image
/// index, as [`fit`](super::fit::fit) takes them. Reading the photographs is
/// the caller's.
///
/// **A point's constraint is honoured.** A held point is left out
/// ([`RefitOutcome::Held`]). A ranged point is fitted as a free one is, except
/// that its coordinate comes from the triangulation's distance rule at its
/// distance from its reference image's camera centre, and it is written ranged
/// as it was ([`RefitRange::Applied`]). A ranged point at a finite distance
/// from an image with no pose is fitted free, and its constraint is released
/// only where the fit makes it a direction ([`RefitRange::Unresolved`]).
///
/// **A point at infinity is fitted as a bearing**, and the classification
/// decides whether it now earns a depth, as *Fit* does; so a free point can
/// cross in either direction ([`RefitPointsReport::crossed`]). Nothing here
/// divides by `w`: a record's columns are copied, and the one number derived
/// from them, a point's shift, is measured only where the point is a position
/// before and after.
///
/// **A point the fit refuses keeps every column it had** and its status names
/// the refusal ([`RefitOutcome::NotFitted`]); so does a point the fit renders
/// no bitmap for in a value that stores one per point
/// ([`RefitOutcome::NoBitmap`]). A point whose record agrees with the stored
/// one is not rewritten, unless its stored reference is a pick only the
/// display render made, which the record writes as the point's own.
///
/// **The answer is the same for any thread count.** Every point is fitted
/// against the value the run was handed, in parallel over the rayon pool, and
/// the records are written at the end, so no point's fit reads another's
/// answer. Each fit is called with `&Progress::none()`.
///
/// `progress` names the stages (`measure noise` where no `sigma_px` is given,
/// `fit points`, which counts the points, and `write back`) and is how the run
/// is asked to stop. The flag is read in front of each stage and of each
/// point's fit; a cancelled run returns [`RefitPointsError::Cancelled`] and
/// writes nothing.
///
/// # Example
///
/// ```no_run
/// use sfmtool_core::bench::{refit_points, RefitPointsOptions};
/// use sfmtool_core::progress::Progress;
/// # fn run(
/// #     recon: &sfmtool_core::SfmrReconstruction,
/// #     views: &[sfmtool_core::patch::normal_refine::ProjectedImage<'_>],
/// # ) -> Result<(), Box<dyn std::error::Error>> {
/// let (next, report) =
///     refit_points(recon, views, &RefitPointsOptions::default(), &Progress::none())?;
/// println!(
///     "{} of {} points refit, {} left as they were",
///     report.refit(),
///     report.read(),
///     report.kept(),
/// );
/// # let _ = next;
/// # Ok(())
/// # }
/// ```
///
/// # Errors
///
/// [`RefitPointsError`] states which precondition of the whole run did not
/// hold, or that the run was cancelled.
pub fn refit_points(
    recon: &SfmrReconstruction,
    images: &[ProjectedImage<'_>],
    options: &RefitPointsOptions,
    progress: &Progress<'_>,
) -> Result<(SfmrReconstruction, RefitPointsReport), RefitPointsError> {
    progress.check_cancel()?;
    check_preconditions(recon, images)?;
    // Each point's fit is where the time goes; the measure is one walk over
    // the observations and the write one over the records.
    let [p_noise, p_fit, p_write] = progress.split([0.05, 0.90, 0.05]);

    // ---- The noise level, once for every point ----
    let sigma_px = stated_or_measured_noise(options.fit.sigma_px, || {
        let _phase = p_noise.phase("measure noise");
        reprojection_noise_level(recon)
    })
    .map_err(RefitPointsError::NoNoiseLevel)?;
    progress.check_cancel()?;

    let fit_options = FitOptions {
        sigma_px: Some(sigma_px),
        ..options.fit.at_patch_resolution(recon)
    };

    // ---- The copy every point's steps read, and the answers ----
    //
    // The copy is the one the write needs, so wrapping it for the per-point
    // steps costs no second copy of the value.
    let edited = EditedReconstruction::new(Arc::new(recon.clone_for_edit()));
    let ranges = point_ranges(&edited.base);
    let total = recon.point_count();
    let fitted: Option<Vec<(RefitPoint, Option<PointRecord>)>> = {
        let mut phase = p_fit.phase("fit points");
        let step = (total / 100).max(1);
        let done = AtomicUsize::new(0);
        let fitted = (0..total)
            .into_par_iter()
            .map(|p| {
                if phase.is_cancelled() {
                    return None;
                }
                let answer = refit_one(
                    &edited,
                    images,
                    p as u32,
                    ranges.get(p).copied().unwrap_or(PointRange::Free),
                    &fit_options,
                    options.max_shift_px,
                );
                let n = done.fetch_add(1, Ordering::Relaxed) + 1;
                if n.is_multiple_of(step) || n == total {
                    phase.count(n as u64, Some(total as u64), "points");
                }
                Some(answer)
            })
            .collect();
        progress_note!(phase, "{total} points");
        fitted
    };
    let fitted = fitted.ok_or(RefitPointsError::Cancelled)?;
    progress.check_cancel()?;

    // ---- The write, into the copy the fits read ----
    let base = Arc::clone(&edited.base);
    drop(edited);
    let mut out = Arc::unwrap_or_clone(base);
    let mut points = Vec::with_capacity(fitted.len());
    {
        let _phase = p_write.phase("write back");
        let mut writes = Vec::new();
        for (status, record) in fitted {
            if let Some(record) = record {
                writes.push((status.index as usize, record));
            }
            points.push(status);
        }
        write_in_place(&mut out, writes);
    }
    let report = RefitPointsReport { points };
    progress_info!(
        progress,
        "{} of {} points refit, {} unchanged, {} left as they were, {} held",
        report.refit(),
        report.read(),
        report.unchanged(),
        report.kept(),
        report.held()
    );
    Ok((out, report))
}

/// The refusals of the whole run, read off the value and the views before
/// anything is copied.
fn check_preconditions(
    recon: &SfmrReconstruction,
    images: &[ProjectedImage<'_>],
) -> Result<(), RefitPointsError> {
    if recon.point_set.feature_indexes().is_some() {
        return Err(RefitPointsError::NoKeypoints);
    }
    if recon.point_set.patch_u_halfvec_xyz.is_none() {
        return Err(RefitPointsError::NoPatchFrames);
    }
    if !recon
        .image_set
        .images
        .iter()
        .any(|image| is_posed(&image.quaternion_wxyz, &image.translation_xyz))
    {
        return Err(RefitPointsError::NoPosedImage);
    }
    let expected = recon.image_count();
    if images.len() < expected {
        return Err(RefitPointsError::ViewsMissing {
            got: images.len(),
            expected,
        });
    }
    // Every image a track names is in the image set, so no step indexes past it:
    // the fit's minimum point depth reads each observation's camera centre
    // and the record checks each `in` row's image. The value's builders
    // already refuse such a row; this holds for a caller that wrote the
    // public `point_set.tracks` field itself.
    if let Some(observation) = recon
        .point_set
        .tracks
        .iter()
        .find(|o| o.image_index as usize >= expected)
    {
        return Err(RefitPointsError::ImageOutOfRange {
            point: observation.point_index,
            image: observation.image_index,
            image_count: expected,
        });
    }
    Ok(())
}

/// What a point's constraint asks of its refit, resolved before the loop.
#[derive(Debug, Clone, Copy)]
enum PointRange {
    /// Free: the fit's classification decides.
    Free,
    /// Held: left out.
    Held,
    /// Ranged, with the distance rule's entry for it.
    Applied(PointDistance),
    /// Ranged at a distance with no origin to measure it from: fitted free.
    Unresolved,
}

/// Each point's [`PointRange`], at `recon`'s poses.
///
/// A ranged point at an infinite distance names no reference and needs no
/// origin, so its entry is the infinite one with no origin, which the distance
/// rule answers without reading it. At a finite distance the origin is its
/// reference image's camera centre ([`reference_camera_center`], the rule the
/// retriangulation reads it by); a reference with no pose, or a distance that
/// is not a positive number, leaves the point unresolved.
fn point_ranges(recon: &SfmrReconstruction) -> Vec<PointRange> {
    let Some(columns) = recon.point_set.point_constraints.as_ref() else {
        return vec![PointRange::Free; recon.point_count()];
    };
    let image_set: &ImageSet = &recon.image_set;
    (0..recon.point_count())
        .map(|p| {
            let kind = columns.point_constraints[p];
            let distance = columns.constraint_distances[p];
            let reference = columns.constraint_reference_images[p];
            match kind {
                POINT_CONSTRAINT_HELD => PointRange::Held,
                POINT_CONSTRAINT_RANGED if distance == f64::INFINITY => {
                    PointRange::Applied(PointDistance {
                        distance,
                        origin: [f64::NAN; 3],
                    })
                }
                POINT_CONSTRAINT_RANGED if distance.is_finite() && distance > 0.0 => {
                    match reference_camera_center(image_set, reference) {
                        Some(origin) => PointRange::Applied(PointDistance { distance, origin }),
                        None => PointRange::Unresolved,
                    }
                }
                POINT_CONSTRAINT_RANGED => PointRange::Unresolved,
                _ => PointRange::Free,
            }
        })
        .collect()
}

/// One point's refit: its status, and the record to write where it changed.
fn refit_one(
    edited: &EditedReconstruction,
    images: &[ProjectedImage<'_>],
    index: u32,
    range: PointRange,
    options: &FitOptions,
    max_shift_px: f64,
) -> (RefitPoint, Option<PointRecord>) {
    let status = |outcome| RefitPoint { index, outcome };
    let (entry, applied) = match range {
        PointRange::Held => return (status(RefitOutcome::Held), None),
        PointRange::Free => (None, RefitRange::Free),
        PointRange::Applied(entry) => (Some(entry), RefitRange::Applied),
        PointRange::Unresolved => (None, RefitRange::Unresolved { released: false }),
    };
    let stored = stored_in_record_order(
        edited
            .point(index)
            .expect("every row of a value with no overlay is a live point")
            .to_record(),
    );
    let mut track =
        track_of_point(edited, index, None).expect("every row of the value is a live point");
    track.thresholds.max_shift_px = max_shift_px;

    // ---- The fit, as `fit` runs it at the track stage ----
    if let Err(e) = fit_preconditions(&track) {
        return (status(RefitOutcome::NotFitted(e)), None);
    }
    let patch_frame = match &track.stage {
        Stage::Track(payload) => payload.patch_frame.clone(),
        Stage::Cluster(_) => None,
    }
    .expect("fit_preconditions refuses a track-stage track with no patch frame");
    let (fitted, report) = match fit_track(
        &track,
        edited,
        images,
        &patch_frame,
        options,
        entry,
        &Progress::none(),
    ) {
        Ok(answer) => answer,
        Err(e) => return (status(RefitOutcome::NotFitted(e)), None),
    };

    // ---- The record a commit would write ----
    let mut record = match point_record(edited, &fitted) {
        Ok(record) => record,
        Err(CommitError::NoBitmap) => return (status(RefitOutcome::NoBitmap), None),
        // The run refused a `sift_files` value, and a fitted track is at the
        // track stage with a position, a patch frame, two or more `in` rows and a
        // keypoint on every one of them, each in an image of the value, since
        // the run refused a track naming any other; the record is built, never
        // applied, so no edit refuses it.
        Err(e) => unreachable!("a fitted track's record is refused only for its bitmap: {e}"),
    };
    let at_infinity = record.point.is_at_infinity();
    let range = match applied {
        RefitRange::Unresolved { .. } => RefitRange::Unresolved {
            released: at_infinity,
        },
        other => other,
    };
    // The commit writes every point free; the refit keeps the point's own
    // constraint, except where a finite distance no longer describes it.
    record.constraint = match range {
        RefitRange::Unresolved { released: true } => {
            Some((POINT_CONSTRAINT_FREE, f64::NAN, NO_REFERENCE_IMAGE))
        }
        _ => stored.constraint,
    };

    let was_at_infinity = stored.point.is_at_infinity();
    let crossed = was_at_infinity != at_infinity;
    let shift = (!was_at_infinity && !at_infinity)
        .then(|| (record.point.position - stored.point.position).norm());
    let unchanged = stored.agrees_with(&record) && !stored.display_only_reference;
    (
        status(RefitOutcome::Refit {
            placed: report.placed,
            kept_at_seed: report.kept_at_seed,
            crossed,
            shift,
            classification: report.classification,
            range,
            unchanged,
        }),
        (!unchanged).then_some(record),
    )
}

/// Write each `(row, record)` into `out` at the point's own row.
///
/// A record's observations are the point's own, all `in`, and the commit's
/// record lists them in image order, ties kept in track order. A stored track
/// is in image order when it was read from a file, but a value edited in memory
/// need not be: tracks replaced through `clone_with_changes` from Python keep
/// the order they were given in. So each record observation is written over
/// the stored row it came from, the `k`-th row of the point's stored rows in
/// that same order ([`record_rows`]), and no row moves. A track that names one
/// image twice is written the same way: the tie goes to track order on both
/// sides. The record's reference, an index into its own image-ordered list, is
/// written as the stored row it names. The columns written are those `out`
/// carries, under the commit's rules: a readings row stands only under the
/// column's own options, a value with no readings column gains one where a
/// record brings a measured reading, and a written point's display-only mark is cleared,
/// since its reference is now the record's own. A bitmap column rendered for
/// display keeps its mark ([`PointSet::patch_bitmaps_for_display`]), so a save
/// still writes the columns the file had.
///
/// # Panics
///
/// Panics if a record's images are not its stored track's, or its bitmap is
/// not the column's shape.
fn write_in_place(out: &mut SfmrReconstruction, writes: Vec<(usize, PointRecord)>) {
    let PointSet {
        points,
        tracks,
        observation_offsets,
        observations,
        patch_u_halfvec_xyz,
        patch_v_halfvec_xyz,
        patch_bitmaps_y_x_rgba,
        normal_confidence,
        point_constraints,
        observation_confidence,
        observation_readings,
        reference_observations,
        display_only_references,
        ..
    } = &mut out.point_set;
    let mut keypoints = match observations {
        ObservationSource::EmbeddedPatches { keypoints_xy, .. } => Some(keypoints_xy),
        ObservationSource::SiftFiles { keypoints_xy, .. } => keypoints_xy.as_mut(),
    };
    // The column is shared with the value the run was handed, so it is copied
    // once here, not once per point.
    let mut bitmaps = (!writes.is_empty())
        .then(|| patch_bitmaps_y_x_rgba.take().map(Arc::unwrap_or_clone))
        .flatten();
    // A value with no readings column gains one where a record brings a
    // measured reading, as a run of commits in point order would add it
    // (`EditedReconstruction::push_record`): under the options of the first
    // such record, every row not measured until a record writes it.
    if observation_readings.is_none() {
        *observation_readings = writes
            .iter()
            .find_map(|(_, record)| {
                record
                    .observations
                    .iter()
                    .any(|o| o.reading.is_some_and(|r| r.is_measured()))
                    .then_some(record.reading_options)
                    .flatten()
            })
            .map(|options| ObservationReadings::not_measured(tracks.len(), options));
    }

    for (p, record) in writes {
        let start = observation_offsets[p];
        let rows = record_rows(&tracks[start..observation_offsets[p + 1]]);
        assert!(
            record.observations.len() == rows.len()
                && record
                    .observations
                    .iter()
                    .zip(&rows)
                    .all(|(o, &k)| o.image_index == tracks[start + k].image_index),
            "point {p}'s record observes other images than its stored track"
        );
        for (&k, observation) in rows.iter().zip(&record.observations) {
            let row = start + k;
            if let (Some(kp), Some([u, v])) = (keypoints.as_deref_mut(), observation.keypoint_xy) {
                kp[[row, 0]] = u;
                kp[[row, 1]] = v;
            }
            if let (Some(column), Some(c)) =
                (observation_confidence.as_mut(), observation.confidence)
            {
                column[row] = c;
            }
            if let Some(readings) = observation_readings.as_mut() {
                let same = record.reading_options == Some(readings.options);
                readings.rows[row] = observation
                    .reading
                    .filter(|_| same)
                    .unwrap_or(ObservationReading::NOT_MEASURED);
            }
        }
        if let (Some(column), Some(h)) = (patch_u_halfvec_xyz.as_mut(), record.patch_u_halfvec) {
            for (c, value) in h.into_iter().enumerate() {
                column[[p, c]] = value;
            }
        }
        if let (Some(column), Some(h)) = (patch_v_halfvec_xyz.as_mut(), record.patch_v_halfvec) {
            for (c, value) in h.into_iter().enumerate() {
                column[[p, c]] = value;
            }
        }
        if let (Some(column), Some(bitmap)) = (bitmaps.as_mut(), record.patch_bitmap.as_ref()) {
            let mut row = column.index_axis_mut(Axis(0), p);
            assert_eq!(
                row.shape(),
                bitmap.shape(),
                "point {p}'s bitmap is not the column's shape"
            );
            row.assign(bitmap);
        }
        if let (Some(column), Some(c)) = (normal_confidence.as_mut(), record.normal_confidence) {
            column[p] = c;
        }
        if let (Some(columns), Some((kind, distance, reference))) =
            (point_constraints.as_mut(), record.constraint)
        {
            columns.point_constraints[p] = kind;
            columns.constraint_distances[p] = distance;
            columns.constraint_reference_images[p] = reference;
        }
        if let (Some(column), Some(r)) = (
            reference_observations.as_mut(),
            record.reference_observation,
        ) {
            column[p] = usize::try_from(r)
                .ok()
                .and_then(|r| rows.get(r))
                .map_or(r, |&k| k as i32);
        }
        if let Some(marks) = display_only_references.as_mut() {
            marks[p] = record.display_only_reference;
        }
        points[p] = record.point;
    }
    if let Some(bitmaps) = bitmaps {
        *patch_bitmaps_y_x_rgba = Some(Arc::new(bitmaps));
    }
    // A point that crossed changes the count of points at infinity.
    out.rebuild_derived_fields();
}

/// A point's stored rows, as offsets into its run, in the order the commit's
/// record lists its observations: by image, ties in track order. The bench's
/// rows are the stored rows in order ([`track_of_point`]), and the record sorts
/// them by `(image, row)`, so the record's `k`-th observation came from the
/// stored row at offset `record_rows(..)[k]`.
fn record_rows(track: &[crate::reconstruction::data::TrackObservation]) -> Vec<usize> {
    let mut rows: Vec<usize> = (0..track.len()).collect();
    rows.sort_by_key(|&k| (track[k].image_index, k));
    rows
}

/// A stored point's record with its observations in the order the commit's
/// record lists them ([`record_rows`]) and its reference renumbered to match,
/// so that [`PointRecord::agrees_with`], which compares observations position
/// by position, compares each stored row with the record observation that
/// came from it. A reference that names none of the point's rows is kept as
/// it is ([`PointRecord::into_image_order`]).
pub(super) fn stored_in_record_order(stored: PointRecord) -> PointRecord {
    stored.into_image_order()
}
