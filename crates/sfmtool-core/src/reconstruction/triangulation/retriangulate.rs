// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Re-solving points a reconstruction already holds, from their own
//! observations at its own poses and its own lenses.
//!
//! [`super::points`] reads flat arrays and knows nothing about a
//! reconstruction. This is the one operation a caller holding a value asks for:
//! re-solve these points of it and hand back the value that holds the answers,
//! the map from its indexes to that value's, and a report. Nothing else moves --
//! no camera, no lens, and no point the caller did not name.
//!
//! See `specs/core/reconstruction/triangulation-rules.md` for the design.

use std::sync::Arc;

use nalgebra::Point3;

use sfmtool_sfmr_format::{NO_REFERENCE_IMAGE, POINT_CONSTRAINT_HELD, POINT_CONSTRAINT_RANGED};

use super::points::{
    triangulate_points_through_cameras, FewObservations, ObservationSet, PointCensus,
    PointDistance, PointRules, PointVerdict, TriangulatedPoints,
};
use crate::numeric::median_in_place;
use crate::progress::{Cancelled, Progress};
use crate::progress_info;
use crate::reconstruction::bundle_adjust::{is_posed, patch_frame_factor, rescale_patch_frame};
use crate::reconstruction::data::{ImageTable, Point3D};
use crate::reconstruction::edited::{EditError, EditedReconstruction, PointMap};

/// Which points one retriangulation re-solves.
///
/// The two cases are also the two shapes a point edit takes, and that is not a
/// coincidence: re-solving every point rewrites the whole point list, which is a
/// new base, and re-solving a handful is a delete-and-re-add of their records,
/// which is an overlay edit. So the caller states which points it means and
/// [`retriangulate_points`] produces the edit that fits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RetriangulateWhich<'a> {
    /// Every point the value holds. The overlay is folded in first and the
    /// answer is the next version's base.
    All,
    /// The points these indexes name, in the value's own indexing. An overlay
    /// edit, and so for a handful: each point costs a rebuild of the addition
    /// set's derived indexes.
    These(&'a [u32]),
}

/// The rules a retriangulation judges each point by.
///
/// The same rules [`PointRules`] holds, minus the two a reconstruction answers
/// for itself: the incoming direction mark is the point's own `w`, and the
/// distance rule is the value's own constraint columns.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RetriangulateOptions {
    /// The angular floor, in radians: a point whose widest ray pair subtends
    /// less than this is a direction rather than a position. `None` is off,
    /// which is the default, and off means no free point crosses between the
    /// two representations on the rays' account alone.
    pub floor_rad: Option<f64>,
    /// Demote a point that solves behind a camera observing it. On by default:
    /// a point the cameras that see it stand in front of is not a place in the
    /// scene, and the direction its rays agree on is the most its observations
    /// support.
    pub cheirality: bool,
    /// Read that demotion per observation, so that a point a minority of its
    /// observations disagree with is solved on the majority. On by default, and
    /// read only when [`Self::cheirality`] is on.
    pub prune_behind: bool,
    /// The pixel bound a fresh estimate has to reproject inside of. `None` is
    /// off, which is the default: the operation is asked what the observations
    /// support, and a caller that wants a residual gate states it.
    pub bar_px: Option<f64>,
}

impl Default for RetriangulateOptions {
    fn default() -> Self {
        Self {
            floor_rad: None,
            cheirality: true,
            prune_behind: true,
            bar_px: None,
        }
    }
}

impl RetriangulateOptions {
    /// The per-point rules these options state, with the value's own distance
    /// column plugged in.
    fn rules<'a>(&self, distance: Option<&'a [PointDistance]>) -> PointRules<'a> {
        PointRules {
            distance,
            floor_rad: self.floor_rad,
            cheirality: self.cheirality,
            prune_behind: self.prune_behind,
            bar_px: self.bar_px,
            // A point with fewer than two usable observations comes back absent
            // and keeps the position it had: the operation has nothing to say
            // about it, and saying nothing is not the same as saying it is a
            // direction.
            few: FewObservations::Absent,
        }
    }
}

/// Why a retriangulation produced nothing. Every variant names what did not
/// hold, because the caller is a menu entry that has to say so in one sentence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RetriangulateError {
    /// The observations carry no pixel: a `sift_files` value without the
    /// format's optional inline keypoint column.
    NoKeypoints,
    /// No image of the reconstruction carries a usable pose.
    NoPosedImages,
    /// An index names no live point of the value.
    NoSuchPoint(u32),
    /// Nothing was left to solve: every point named is held, or the value holds
    /// no point at all.
    NothingToSolve,
    /// The caller asked the operation to stop, and it did, so there is no
    /// answer to write back.
    Cancelled,
    /// The overlay refused a record the operation built from one of its own
    /// points. Not reachable from a value whose columns are parallel.
    Edit(EditError),
}

impl std::fmt::Display for RetriangulateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RetriangulateError::NoKeypoints => write!(
                f,
                "retriangulation needs a pixel per observation, and this reconstruction's \
                 observations are .sift feature indexes with no inline keypoints"
            ),
            RetriangulateError::NoPosedImages => {
                write!(f, "no image of this reconstruction carries a pose")
            }
            RetriangulateError::NoSuchPoint(index) => {
                write!(f, "no live point at index {index}")
            }
            RetriangulateError::NothingToSolve => write!(
                f,
                "nothing was left to retriangulate: every point named is held at the \
                 coordinate it has"
            ),
            RetriangulateError::Cancelled => write!(
                f,
                "the retriangulation was asked to stop before it had an answer, so nothing \
                 was written back"
            ),
            RetriangulateError::Edit(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for RetriangulateError {}

impl From<Cancelled> for RetriangulateError {
    fn from(_: Cancelled) -> Self {
        RetriangulateError::Cancelled
    }
}

impl From<EditError> for RetriangulateError {
    fn from(e: EditError) -> Self {
        RetriangulateError::Edit(e)
    }
}

/// What one retriangulation did, point by point.
///
/// [`Self::points`] is the record, and every count the report gives is read
/// off it, so a count cannot disagree with the statuses it summarises. The
/// census is the array solve's own, over the points that were read, and it
/// agrees with the statuses as well: its `seen` is [`Self::read`], its `few`
/// is [`Self::kept`], each of its other buckets is
/// [`Self::with_verdict`] of that bucket's verdict, and its `pruned_obs` is the
/// sum of the statuses' `pruned`.
#[derive(Debug, Clone, PartialEq)]
pub struct RetriangulateReport {
    /// One status per point the call was asked about, held points included, in
    /// ascending order of [`RetriangulatedPoint::index`]. Under
    /// [`RetriangulateWhich::All`] that is every live point of the value; under
    /// [`RetriangulateWhich::These`] it is each index named, once.
    pub points: Vec<RetriangulatedPoint>,
    /// Observations behind the points that were read.
    pub observations: usize,
    /// How many points each rule decided, and what the finite ones look like.
    pub census: PointCensus,
}

/// What one retriangulation did to one point, and where to find it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RetriangulatedPoint {
    /// The point's index in the value the call was given.
    pub index: u32,
    /// The point's index in the value the call returned, which is where the
    /// returned [`PointMap`] forwards [`Self::index`] to. A point
    /// [`RetriangulateWhich::These`] rewrote takes a new index; a point it did
    /// not rewrite keeps the one it had, and under [`RetriangulateWhich::All`]
    /// this is the point's row in the new base.
    pub new_index: u32,
    /// What happened to it.
    pub outcome: RetriangulateOutcome,
}

/// What a retriangulation did to one point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RetriangulateOutcome {
    /// The value holds the point at the coordinate it has, so the solve never
    /// read it and its geometry is unchanged.
    Held,
    /// Fewer than two of its observations state a usable ray, which is the
    /// solve's [`PointVerdict::Few`]. The operation has nothing to say about
    /// the point, so it keeps the geometry it had.
    Kept,
    /// The solve answered, and the answer is what the point now holds.
    Solved {
        /// The rule that decided it, never [`PointVerdict::Few`]. Under
        /// [`PointVerdict::Finite`], [`PointVerdict::FinitePruned`] and
        /// [`PointVerdict::Ranged`] at a finite distance the answer is a
        /// position; under [`PointVerdict::Marked`], [`PointVerdict::Thin`],
        /// [`PointVerdict::Behind`], [`PointVerdict::OverBar`] and
        /// [`PointVerdict::Ranged`] at an infinite distance it is a direction.
        verdict: PointVerdict,
        /// Observations the cheirality prune left out of the solve because they
        /// see the point behind them. Nonzero only under
        /// [`PointVerdict::FinitePruned`]. They stay on the point's track: the
        /// prune leaves them out of this solve, and deleting them is a separate
        /// edit.
        pruned: u32,
        /// How the answer differs from the geometry the point had.
        change: GeometryChange,
    },
}

/// How a solved point's answer differs from the geometry it had.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GeometryChange {
    /// The answer is the stored geometry, bit for bit, so nothing was written.
    Unchanged,
    /// A position that stayed a position, or a direction that stayed a
    /// direction, moved.
    Moved {
        /// How far a position travelled, in the value's own units; `None` for a
        /// direction, whose change is a turn and not a distance.
        shift: Option<f64>,
    },
    /// A position became a direction, or a direction became a position.
    Crossed,
}

impl RetriangulateOutcome {
    /// The verdict the solve reached, or `None` for a point it never read.
    /// [`Self::Kept`] is [`PointVerdict::Few`].
    pub fn verdict(&self) -> Option<PointVerdict> {
        match self {
            RetriangulateOutcome::Held => None,
            RetriangulateOutcome::Kept => Some(PointVerdict::Few),
            RetriangulateOutcome::Solved { verdict, .. } => Some(*verdict),
        }
    }

    /// Whether the point's stored geometry changed.
    pub fn moved(&self) -> bool {
        matches!(
            self,
            RetriangulateOutcome::Solved {
                change: GeometryChange::Moved { .. } | GeometryChange::Crossed,
                ..
            }
        )
    }
}

/// The outcome in the words a reader is shown, as the tail of a sentence about
/// one point: "Retriangulated point 42 in run_a: `finite, unchanged`".
///
/// The verdict's own [`PointVerdict::label`], then how many observations the
/// prune left out where it left any out, then `unchanged` where the answer is
/// the geometry the point already had. Held here so the viewer's Action Log and
/// the wire say the same words about the same point.
impl std::fmt::Display for RetriangulateOutcome {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RetriangulateOutcome::Held => write!(f, "held at its coordinate, so not read"),
            RetriangulateOutcome::Kept => f.write_str(PointVerdict::Few.label()),
            RetriangulateOutcome::Solved {
                verdict,
                pruned,
                change,
            } => {
                f.write_str(verdict.label())?;
                match pruned {
                    0 => {}
                    1 => f.write_str(" (1 observation that sees it behind was left out)")?,
                    n => write!(f, " ({n} observations that see it behind were left out)")?,
                }
                if *change == GeometryChange::Unchanged {
                    f.write_str(", unchanged")?;
                }
                Ok(())
            }
        }
    }
}

impl RetriangulateReport {
    /// The status of the point `index` names in the value the call was given,
    /// or `None` where the call was not asked about it.
    pub fn point(&self, index: u32) -> Option<&RetriangulatedPoint> {
        self.points
            .binary_search_by_key(&index, |p| p.index)
            .ok()
            .map(|k| &self.points[k])
    }

    /// Points the solve read: every status but [`RetriangulateOutcome::Held`].
    pub fn read(&self) -> usize {
        self.count(|o| !matches!(o, RetriangulateOutcome::Held))
    }

    /// Points the solve never read, because the value holds them at the
    /// coordinate they have.
    pub fn held(&self) -> usize {
        self.count(|o| matches!(o, RetriangulateOutcome::Held))
    }

    /// Points whose stored geometry the answer replaced.
    pub fn moved(&self) -> usize {
        self.count(RetriangulateOutcome::moved)
    }

    /// Of those, the ones that changed representation: a position that became a
    /// direction, or a direction that became a position.
    pub fn crossed(&self) -> usize {
        self.count(|o| {
            matches!(
                o,
                RetriangulateOutcome::Solved {
                    change: GeometryChange::Crossed,
                    ..
                }
            )
        })
    }

    /// Points the operation had nothing to say about, which keep the geometry
    /// they had: fewer than two of their observations state a usable ray.
    pub fn kept(&self) -> usize {
        self.count(|o| matches!(o, RetriangulateOutcome::Kept))
    }

    /// Points the solve gave `verdict`. [`PointVerdict::Few`] counts the
    /// [`Self::kept`] points.
    pub fn with_verdict(&self, verdict: PointVerdict) -> usize {
        self.count(|o| o.verdict() == Some(verdict))
    }

    /// Median distance a moved finite point travelled, in the value's own
    /// units; `NaN` where no finite point moved.
    ///
    /// Over the points that were finite before and after alone, because a
    /// crossing has no distance: the value before and the value after are a
    /// place and a direction, and what subtracting one from the other produces
    /// is not a distance in the scene.
    pub fn median_shift(&self) -> f64 {
        let mut shifts: Vec<f64> = self
            .points
            .iter()
            .filter_map(|p| match p.outcome {
                RetriangulateOutcome::Solved {
                    change: GeometryChange::Moved { shift },
                    ..
                } => shift,
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
    fn count(&self, keep: impl Fn(&RetriangulateOutcome) -> bool) -> usize {
        self.points.iter().filter(|p| keep(&p.outcome)).count()
    }
}

/// Re-solve the points `which` names from their own observations, at `edited`'s
/// poses and lenses, and hand back the value that holds the answers.
///
/// This is the observation form of the triangulation rules over a
/// reconstruction: it gathers the pixels, the poses, the camera each image was
/// taken through and the constraint columns the array form takes, runs it
/// once, and writes each point's answer back. Each observation's ray is cast,
/// and its reprojection read, through its own image's camera, so a value whose
/// images are taken through several cameras -- a rig with one camera per
/// sensor -- is solved in the same call. It is pure -- `edited` is left exactly
/// as it was -- and it moves no camera and no lens. What a point's observations
/// support at *this* geometry is the whole of what it decides.
///
/// **A point's constraint is honoured.** A held point is never read: the value
/// owns its coordinate, so it is not in the solve and not in the answer. A
/// ranged point keeps its distance and only its direction is re-read, measured
/// from its reference image's camera centre at these poses; a ranged point
/// whose reference the value does not name, or does not pose, is solved free,
/// because a distance from nothing constrains nothing.
///
/// **A point the operation cannot speak for keeps the geometry it has.** That
/// is a point fewer than two of whose observations state a usable ray, which
/// comes back absent; its status is [`RetriangulateOutcome::Kept`] rather than
/// a `NaN` written into the value. Every other verdict is written: a point the
/// floor calls thin, or cheirality refuses, or the bar turns down, becomes the
/// direction its rays agree on, and the patch frame of a point that moved is
/// rescaled so the patch keeps the angular size it had.
///
/// **The report says what happened to every point asked about.**
/// [`RetriangulateReport::points`] holds one [`RetriangulatedPoint`] per point,
/// held ones included, carrying its index in `edited`, its index in the value
/// returned, and its [`RetriangulateOutcome`]: held and not read, kept for too
/// few observations, or solved under a named verdict, with the observations the
/// cheirality prune left out and how the geometry changed. The report's counts
/// are read off those statuses.
///
/// `progress` names the three stages (gathering the arrays, the solve, writing
/// the answer back) and is how the call is asked to stop. A cancel is read
/// between stages: a stopped call returns [`RetriangulateError::Cancelled`] and
/// writes nothing, because a value whose points were half re-solved is not an
/// answer anybody asked for. Pass `&Progress::none()` to report nothing and
/// never stop.
///
/// # Example
///
/// ```no_run
/// use sfmtool_core::progress::Progress;
/// use sfmtool_core::reconstruction::triangulation::{
///     retriangulate_points, RetriangulateOptions, RetriangulateWhich,
/// };
/// # fn run(edited: &sfmtool_core::EditedReconstruction)
/// # -> Result<(), Box<dyn std::error::Error>> {
/// let (next, map, report) = retriangulate_points(
///     edited,
///     RetriangulateWhich::All,
///     &RetriangulateOptions::default(),
///     &Progress::none(),
/// )?;
/// println!("{} of {} points moved", report.moved(), report.read());
/// for point in &report.points {
///     if !point.outcome.moved() {
///         println!("point {} (now {}): {}", point.index, point.new_index, point.outcome);
///     }
/// }
/// # let _ = (next, map);
/// # Ok(())
/// # }
/// ```
///
/// # Errors
///
/// [`RetriangulateError`] states which precondition did not hold, or that the
/// call was cancelled.
pub fn retriangulate_points(
    edited: &EditedReconstruction,
    which: RetriangulateWhich<'_>,
    options: &RetriangulateOptions,
    progress: &Progress<'_>,
) -> Result<(EditedReconstruction, PointMap, RetriangulateReport), RetriangulateError> {
    progress.check_cancel()?;
    // The solve is where the time goes; the other two walk arrays the size of
    // what was asked for. An estimate, as every set of weights is.
    let [p_gather, p_solve, p_write] = progress.split([0.20, 0.65, 0.15]);

    // `All` rewrites the whole point list, so the overlay is folded in first and
    // the answer becomes the next version's base; `These` writes the overlay and
    // leaves the base alone, which is what keeps every other index meaning what
    // it meant.
    let fold = matches!(which, RetriangulateWhich::All)
        && !(edited.deleted_points.is_empty() && edited.added.points.is_empty());
    let (work, folded) = if fold {
        let _phase = p_gather.phase("materialise");
        let (value, map) = edited.materialize();
        (
            EditedReconstruction::new(Arc::new(value)),
            Some(PointMap::Rows(map)),
        )
    } else {
        (edited.clone(), None)
    };

    let gathered = {
        let mut phase = p_gather.phase("gather observations");
        let gathered = gather(&work, which)?;
        crate::progress_note!(
            phase,
            "{} points, {} observations",
            gathered.targets.len(),
            gathered.obs_point.len()
        );
        gathered
    };
    progress.check_cancel()?;

    let table = &work.base.image_table;
    let estimates = {
        let _phase = p_solve.phase("retriangulate");
        triangulate_points_through_cameras(
            &table.cameras,
            &gathered.image_camera,
            ObservationSet {
                uv: &gathered.uv,
                obs_image: &gathered.obs_image,
                obs_point: &gathered.obs_point,
                quats_wxyz: &gathered.quats_wxyz,
                translations: &gathered.translations,
                n_tracks: gathered.targets.len(),
            },
            Some(&gathered.marks),
            options.rules(gathered.distance.as_deref()),
            None,
        )
    };
    progress.check_cancel()?;

    let (next, map, mut points) = {
        let _phase = p_write.phase("write back");
        match which {
            RetriangulateWhich::All => write_whole_value(&work, &gathered, &estimates),
            RetriangulateWhich::These(_) => write_overlay(edited, &gathered, &estimates)?,
        }
    };
    // A held point is not rewritten by either write, so it keeps the index it
    // has in `work`: the same row of the new base under `All`, the same index
    // of the overlay under `These`.
    points.extend(gathered.held.iter().map(|&index| RetriangulatedPoint {
        index,
        new_index: index,
        outcome: RetriangulateOutcome::Held,
    }));
    let map = match folded {
        Some(materialised) => {
            // Every status was written in `work`'s indexing, which is the
            // materialisation's; the caller holds `edited`'s, so each is read
            // back through the fold. Every row of a materialisation has a
            // source, and the fold need not keep the order, so the statuses are
            // sorted after.
            for point in &mut points {
                point.index = materialised
                    .inverse(point.index)
                    .expect("every materialised row has a source");
            }
            PointMap::Chain(vec![materialised, map])
        }
        None => map,
    };
    points.sort_unstable_by_key(|p| p.index);
    let report = RetriangulateReport {
        points,
        observations: gathered.obs_point.len(),
        census: estimates.census,
    };
    progress_info!(
        progress,
        "{} of {} points moved, {} kept, {} held",
        report.moved(),
        report.read(),
        report.kept(),
        report.held()
    );
    Ok((next, map, report))
}

/// The arrays the array form takes, plus what the write-back needs to read them
/// back against.
struct Gathered {
    /// The live indexes the solve is over, ascending, one per track slot.
    targets: Vec<u32>,
    /// The live indexes not in `targets` because the value holds their
    /// coordinate, ascending.
    held: Vec<u32>,
    /// Per image, the camera-table index of the camera it was taken through.
    image_camera: Vec<u32>,
    uv: Vec<f64>,
    obs_image: Vec<u32>,
    obs_point: Vec<u32>,
    quats_wxyz: Vec<f64>,
    translations: Vec<f64>,
    /// Per target, whether the value carries it as a direction.
    marks: Vec<bool>,
    /// Per target, the distance rule its constraint states; `None` where the
    /// value constrains no point's distance.
    distance: Option<Vec<PointDistance>>,
}

/// Read `work`'s poses, pixels and constraints into the arrays the array form
/// takes.
fn gather(
    work: &EditedReconstruction,
    which: RetriangulateWhich<'_>,
) -> Result<Gathered, RetriangulateError> {
    if !work.has_keypoints() {
        return Err(RetriangulateError::NoKeypoints);
    }
    let table = &work.base.image_table;

    let posed: Vec<usize> = (0..table.images.len())
        .filter(|&i| {
            is_posed(
                &table.images[i].quaternion_wxyz,
                &table.images[i].translation_xyz,
            )
        })
        .collect();
    if posed.is_empty() {
        return Err(RetriangulateError::NoPosedImages);
    }
    // Each image names the camera that took it, and the solve casts an
    // observation's ray, and reads its reprojection, through that camera.
    let image_camera: Vec<u32> = table.images.iter().map(|i| i.camera_index).collect();

    // Every image gets a row, so an observation indexes the table directly. An
    // unposed image's row is not a pose, and the array form drops the rays it
    // would have cast rather than casting them through a placeholder.
    let mut quats_wxyz = Vec::with_capacity(table.images.len() * 4);
    let mut translations = Vec::with_capacity(table.images.len() * 3);
    for (i, image) in table.images.iter().enumerate() {
        if posed.binary_search(&i).is_ok() {
            let q = image.quaternion_wxyz;
            quats_wxyz.extend_from_slice(&[q.w, q.i, q.j, q.k]);
            let t = image.translation_xyz;
            translations.extend_from_slice(&[t.x, t.y, t.z]);
        } else {
            quats_wxyz.extend_from_slice(&[f64::NAN; 4]);
            translations.extend_from_slice(&[f64::NAN; 3]);
        }
    }

    // The candidates, then the held ones taken out of them: a held point is not
    // a point the solve declines to move, it is a point the solve never sees.
    let candidates: Vec<u32> = match which {
        RetriangulateWhich::All => work.live_indexes().collect(),
        RetriangulateWhich::These(indexes) => {
            for &index in indexes {
                if work.point(index).is_none() {
                    return Err(RetriangulateError::NoSuchPoint(index));
                }
            }
            let mut these = indexes.to_vec();
            these.sort_unstable();
            these.dedup();
            these
        }
    };
    let mut held = Vec::new();
    let mut targets = Vec::with_capacity(candidates.len());
    for index in candidates {
        let view = work.point(index).expect("a live index");
        if view
            .constraint()
            .is_some_and(|(k, _, _)| k == POINT_CONSTRAINT_HELD)
        {
            held.push(index);
            continue;
        }
        targets.push(index);
    }
    if targets.is_empty() {
        return Err(RetriangulateError::NothingToSolve);
    }

    let mut uv = Vec::new();
    let mut obs_image = Vec::new();
    let mut obs_point = Vec::new();
    let mut marks = Vec::with_capacity(targets.len());
    let mut distance = Vec::with_capacity(targets.len());
    let mut any_ranged = false;
    for (slot, &index) in targets.iter().enumerate() {
        let view = work.point(index).expect("a live index");
        marks.push(view.point().is_at_infinity());
        distance.push(ranged_entry(table, view.constraint(), &mut any_ranged));
        for (k, observation) in view.observations().iter().enumerate() {
            let Some([x, y]) = view.keypoint_xy(k) else {
                continue;
            };
            uv.push(f64::from(x));
            uv.push(f64::from(y));
            obs_image.push(observation.image_index);
            obs_point.push(slot as u32);
        }
    }
    Ok(Gathered {
        targets,
        held,
        image_camera,
        uv,
        obs_image,
        obs_point,
        quats_wxyz,
        translations,
        marks,
        distance: any_ranged.then_some(distance),
    })
}

/// The distance rule one point's constraint triple states, at these poses.
///
/// A ranged point's origin is a function of the poses and is resolved here, at
/// the geometry being solved under, exactly as the adjustment resolves its own.
/// A point whose reference the value does not name carries no origin to measure
/// from, so the rule says nothing about it and it is solved free.
fn ranged_entry(
    table: &ImageTable,
    constraint: Option<(u8, f64, u32)>,
    any_ranged: &mut bool,
) -> PointDistance {
    let Some((kind, distance, reference)) = constraint else {
        return PointDistance::NONE;
    };
    if kind != POINT_CONSTRAINT_RANGED || reference == NO_REFERENCE_IMAGE {
        return PointDistance::NONE;
    }
    let Some(image) = table.images.get(reference as usize) else {
        return PointDistance::NONE;
    };
    if !is_posed(&image.quaternion_wxyz, &image.translation_xyz) {
        return PointDistance::NONE;
    }
    *any_ranged = true;
    let origin = image.camera_center();
    PointDistance {
        distance,
        origin: [origin.x, origin.y, origin.z],
    }
}

/// One point's answer, read against the geometry it came in with: its outcome,
/// and the geometry to write where that changed.
struct Decision {
    outcome: RetriangulateOutcome,
    /// Where the answer puts the point, and its `w`; `None` where nothing is to
    /// be written.
    write: Option<(Point3<f64>, f64)>,
}

/// Read the estimate of track slot `slot` against `before`, the geometry the
/// point came in with.
fn decide_point(
    before: &Point3D,
    estimates: &TriangulatedPoints,
    pruned: &[u32],
    slot: usize,
) -> Decision {
    let xyzw = estimates.xyzw[slot];
    let verdict = estimates.verdicts[slot];
    // The rules hand `Few` back as an absent estimate, and it is the only
    // verdict that comes back absent: every other one is a solve over two or
    // more finite rays, or their mean. The test is on the estimate itself, so
    // no `NaN` is ever written whatever verdict it came under.
    if !xyzw.iter().all(|c| c.is_finite()) {
        debug_assert_eq!(verdict, PointVerdict::Few);
        return Decision {
            outcome: RetriangulateOutcome::Kept,
            write: None,
        };
    }
    let position = Point3::new(xyzw[0], xyzw[1], xyzw[2]);
    let w = xyzw[3];
    let was_at_infinity = before.is_at_infinity();
    let now_at_infinity = w == 0.0;
    let change = if was_at_infinity != now_at_infinity {
        GeometryChange::Crossed
    } else if position == before.position {
        GeometryChange::Unchanged
    } else {
        GeometryChange::Moved {
            shift: (!now_at_infinity).then(|| (position.coords - before.position.coords).norm()),
        }
    };
    Decision {
        outcome: RetriangulateOutcome::Solved {
            verdict,
            pruned: pruned[slot],
            change,
        },
        write: (change != GeometryChange::Unchanged).then_some((position, w)),
    }
}

/// Per track slot, how many of its observations the cheirality prune left out.
fn pruned_per_slot(gathered: &Gathered, estimates: &TriangulatedPoints) -> Vec<u32> {
    let mut counts = vec![0u32; gathered.targets.len()];
    for (&slot, &dropped) in gathered.obs_point.iter().zip(&estimates.pruned) {
        counts[slot as usize] += u32::from(dropped);
    }
    counts
}

/// The answers written into a fresh base, for a retriangulation of every point,
/// with one status per target in `work`'s indexing.
///
/// `work` holds no overlay here -- it is either the caller's value with an empty
/// one or the materialisation of the value that had one -- so the targets are
/// the base's own rows and the write is in place.
fn write_whole_value(
    work: &EditedReconstruction,
    gathered: &Gathered,
    estimates: &TriangulatedPoints,
) -> (EditedReconstruction, PointMap, Vec<RetriangulatedPoint>) {
    let mut out = work.base.clone_for_edit();
    let pruned = pruned_per_slot(gathered, estimates);
    let mut points = Vec::with_capacity(gathered.targets.len() + gathered.held.len());
    for (slot, &index) in gathered.targets.iter().enumerate() {
        let p = index as usize;
        let decision = decide_point(&out.point_set.points[p], estimates, &pruned, slot);
        // Every point keeps the row it had: this edit deletes none and creates
        // none, it moves them.
        points.push(RetriangulatedPoint {
            index,
            new_index: index,
            outcome: decision.outcome,
        });
        let Some((position, w)) = decision.write else {
            continue;
        };
        let before = &out.point_set.points[p];
        let before_scale = (!before.is_at_infinity()).then(|| {
            work.base
                .image_table
                .placement_scale(&before.position.clone())
        });
        let after = (w != 0.0).then_some(&position);
        rescale_patch_frame(&mut out, p, before_scale, after);
        let point = &mut out.point_set.points[p];
        point.position = position;
        point.w = w;
    }
    out.rebuild_derived_fields();
    (
        EditedReconstruction::new(Arc::new(out)),
        PointMap::Removed(Vec::new()),
        points,
    )
}

/// The answers written as an overlay edit, for a retriangulation of a handful of
/// points, with one status per target.
///
/// Delete-and-re-add, which is what a modified point is here, so each rewritten
/// point takes a new index, and both the map and its status say where it went.
fn write_overlay(
    edited: &EditedReconstruction,
    gathered: &Gathered,
    estimates: &TriangulatedPoints,
) -> Result<(EditedReconstruction, PointMap, Vec<RetriangulatedPoint>), RetriangulateError> {
    let mut next = edited.clone();
    let pruned = pruned_per_slot(gathered, estimates);
    let mut points = Vec::with_capacity(gathered.targets.len() + gathered.held.len());
    let mut moves = Vec::new();
    for (slot, &index) in gathered.targets.iter().enumerate() {
        let view = next.point(index).expect("a live index");
        let mut record = view.to_record();
        let decision = decide_point(&record.point, estimates, &pruned, slot);
        let Some((position, w)) = decision.write else {
            points.push(RetriangulatedPoint {
                index,
                new_index: index,
                outcome: decision.outcome,
            });
            continue;
        };
        let before_scale = (!record.point.is_at_infinity()).then(|| {
            edited
                .base
                .image_table
                .placement_scale(&record.point.position)
        });
        let after = (w != 0.0).then_some(&position);
        if let Some(factor) = patch_frame_factor(&edited.base.image_table, before_scale, after) {
            for half in [&mut record.patch_u_halfvec, &mut record.patch_v_halfvec] {
                if let Some(v) = half.as_mut() {
                    for c in v.iter_mut() {
                        *c *= factor;
                    }
                }
            }
        }
        record.point.position = position;
        record.point.w = w;
        let to = next.replace_point(index, record)?;
        moves.push((index, to));
        points.push(RetriangulatedPoint {
            index,
            new_index: to,
            outcome: decision.outcome,
        });
    }
    Ok((next, PointMap::Replaced(moves), points))
}

#[cfg(test)]
mod tests;
