// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Re-solving points a reconstruction already holds, from their own
//! observations at its own poses and its own lens.
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
    triangulate_points_from_observations, FewObservations, ObservationSet, PointCensus,
    PointDistance, PointRules, TriangulatedPoints,
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
    /// The posed images do not share one set of camera intrinsics.
    MixedCameras {
        /// How many the posed images between them name.
        cameras: usize,
    },
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
            RetriangulateError::MixedCameras { cameras } => write!(
                f,
                "retriangulation reads one shared camera, and these images are taken \
                 through {cameras}"
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

/// What one retriangulation did.
#[derive(Debug, Clone, PartialEq)]
pub struct RetriangulateReport {
    /// Points the operation solved.
    pub read: usize,
    /// Observations behind them.
    pub observations: usize,
    /// Points it never read, because the value holds them at the coordinate
    /// they have.
    pub held: usize,
    /// Points whose stored geometry the answer replaced.
    pub moved: usize,
    /// Of those, the ones that changed representation: a position that became a
    /// direction, or a direction that became a position.
    pub crossed: usize,
    /// Points the operation had nothing to say about, which keep the geometry
    /// they had: fewer than two of their observations state a usable ray.
    pub kept: usize,
    /// How many points each rule decided, and what the finite ones look like.
    pub census: PointCensus,
    /// Median distance a moved finite point travelled, in the value's own
    /// units; `NaN` where no finite point moved.
    ///
    /// Over the points that were finite before and after alone, because a
    /// crossing has no distance: the value before and the value after are a
    /// place and a direction, and what subtracting one from the other produces
    /// is not a distance in the scene.
    pub median_shift: f64,
}

/// Re-solve the points `which` names from their own observations, at `edited`'s
/// poses and lens, and hand back the value that holds the answers.
///
/// This is [`triangulate_points_from_observations`] over a reconstruction: it
/// gathers the pixels, the poses and the constraint columns the array form
/// takes, runs it once, and writes each point's answer back. It is pure --
/// `edited` is left exactly as it was -- and it moves no camera and no lens.
/// What a point's observations support at *this* geometry is the whole of what
/// it decides.
///
/// **A point's constraint is honoured.** A held point is never read: the value
/// owns its coordinate, so it is not in the solve and not in the answer. A
/// ranged point keeps its distance and only its direction is re-read, measured
/// from the mean of its reference images' camera centres at these poses; a
/// ranged point whose reference the value does not name is solved free, because
/// a distance from nothing constrains nothing.
///
/// **A point the operation cannot speak for keeps the geometry it has.** That
/// is a point fewer than two of whose observations state a usable ray, which
/// comes back absent; it is counted in [`RetriangulateReport::kept`] rather
/// than written as a `NaN`. Every other verdict is written: a point the floor
/// calls thin, or cheirality refuses, or the bar turns down, becomes the
/// direction its rays agree on, and the patch frame of a point that moved is
/// rescaled so the patch keeps the angular size it had.
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
/// println!("{} of {} points moved", report.moved, report.read);
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
        triangulate_points_from_observations(
            table.camera_for_image(gathered.camera_image),
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
        )
    };
    progress.check_cancel()?;

    let (next, map, report) = {
        let _phase = p_write.phase("write back");
        match which {
            RetriangulateWhich::All => write_whole_value(&work, &gathered, &estimates),
            RetriangulateWhich::These(_) => write_overlay(edited, &gathered, &estimates)?,
        }
    };
    let map = match folded {
        Some(materialised) => PointMap::Chain(vec![materialised, map]),
        None => map,
    };
    progress_info!(
        progress,
        "{} of {} points moved, {} kept, {} held",
        report.moved,
        report.read,
        report.kept,
        report.held
    );
    Ok((next, map, report))
}

/// The arrays the array form takes, plus what the write-back needs to read them
/// back against.
struct Gathered {
    /// The live indexes the solve is over, ascending, one per track slot.
    targets: Vec<u32>,
    /// Points not in `targets` because the value holds their coordinate.
    held: usize,
    /// An image taken through the one camera the solve reads, for the lookup.
    camera_image: usize,
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

    // One camera for the whole solve: the array form carries a single shared
    // model, and a value whose images disagree about the lens has to be told so
    // rather than silently solved through one of them. An unposed image states
    // no ray, so it is not in the solve and its lens is not in this question.
    let posed: Vec<usize> = (0..table.images.len())
        .filter(|&i| {
            is_posed(
                &table.images[i].quaternion_wxyz,
                &table.images[i].translation_xyz,
            )
        })
        .collect();
    let Some(&camera_image) = posed.first() else {
        return Err(RetriangulateError::NoPosedImages);
    };
    let mut lenses: Vec<u32> = posed
        .iter()
        .map(|&i| table.images[i].camera_index)
        .collect();
    lenses.sort_unstable();
    lenses.dedup();
    if lenses.len() != 1 {
        return Err(RetriangulateError::MixedCameras {
            cameras: lenses.len(),
        });
    }

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
    let mut held = 0usize;
    let mut targets = Vec::with_capacity(candidates.len());
    for index in candidates {
        let view = work.point(index).expect("a live index");
        if view
            .constraint()
            .is_some_and(|(k, _, _)| k == POINT_CONSTRAINT_HELD)
        {
            held += 1;
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
        camera_image,
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

/// What one point's answer is, read against the geometry it had.
struct Decision {
    /// Where the answer puts it, and whether that is a position.
    position: Point3<f64>,
    w: f64,
    /// Whether the stored geometry changed at all.
    moved: bool,
    /// Whether the representation changed with it.
    crossed: bool,
    /// How far a finite point that stayed finite travelled; `None` otherwise.
    shift: Option<f64>,
}

/// Read one point's estimate against the geometry it came in with, or `None`
/// where the operation had nothing to say about it.
fn decide_point(before: &Point3D, xyzw: [f64; 4]) -> Option<Decision> {
    if !xyzw.iter().all(|c| c.is_finite()) {
        return None;
    }
    let position = Point3::new(xyzw[0], xyzw[1], xyzw[2]);
    let w = xyzw[3];
    let was_at_infinity = before.is_at_infinity();
    let now_at_infinity = w == 0.0;
    let crossed = was_at_infinity != now_at_infinity;
    let shift =
        (!crossed && !now_at_infinity).then(|| (position.coords - before.position.coords).norm());
    Some(Decision {
        moved: crossed || position != before.position,
        crossed,
        shift,
        position,
        w,
    })
}

/// The answers written into a fresh base, for a retriangulation of every point.
///
/// `work` holds no overlay here -- it is either the caller's value with an empty
/// one or the materialisation of the value that had one -- so the targets are
/// the base's own rows and the write is in place.
fn write_whole_value(
    work: &EditedReconstruction,
    gathered: &Gathered,
    estimates: &TriangulatedPoints,
) -> (EditedReconstruction, PointMap, RetriangulateReport) {
    let mut out = work.base.clone_for_edit();
    let mut report = empty_report(gathered, estimates);
    let mut shifts: Vec<f64> = Vec::new();
    for (slot, &index) in gathered.targets.iter().enumerate() {
        let p = index as usize;
        let Some(decision) = decide_point(&out.point_set.points[p], estimates.xyzw[slot]) else {
            report.kept += 1;
            continue;
        };
        if !decision.moved {
            continue;
        }
        let before = &out.point_set.points[p];
        let was_at_infinity = before.is_at_infinity();
        let before_scale = (!was_at_infinity).then(|| {
            work.base
                .image_table
                .placement_scale(&before.position.clone())
        });
        let after = (decision.w != 0.0).then_some(&decision.position);
        rescale_patch_frame(&mut out, p, before_scale, after);
        let point = &mut out.point_set.points[p];
        point.position = decision.position;
        point.w = decision.w;
        report.moved += 1;
        report.crossed += usize::from(decision.crossed);
        shifts.extend(decision.shift);
    }
    out.rebuild_derived_fields();
    // The middle of what the finite points travelled; see the field's own doc
    // for the population it is over.
    report.median_shift = if shifts.is_empty() {
        f64::NAN
    } else {
        median_in_place(&mut shifts)
    };
    // Every point keeps the index it had: this edit deletes none and creates
    // none, it moves them.
    (
        EditedReconstruction::new(Arc::new(out)),
        PointMap::Removed(Vec::new()),
        report,
    )
}

/// The answers written as an overlay edit, for a retriangulation of a handful of
/// points.
///
/// Delete-and-re-add, which is what a modified point is here, so each rewritten
/// point takes a new index and the map says where it went.
fn write_overlay(
    edited: &EditedReconstruction,
    gathered: &Gathered,
    estimates: &TriangulatedPoints,
) -> Result<(EditedReconstruction, PointMap, RetriangulateReport), RetriangulateError> {
    let mut next = edited.clone();
    let mut report = empty_report(gathered, estimates);
    let mut shifts: Vec<f64> = Vec::new();
    let mut moves = Vec::new();
    for (slot, &index) in gathered.targets.iter().enumerate() {
        let view = next.point(index).expect("a live index");
        let mut record = view.to_record();
        let Some(decision) = decide_point(&record.point, estimates.xyzw[slot]) else {
            report.kept += 1;
            continue;
        };
        if !decision.moved {
            continue;
        }
        let was_at_infinity = record.point.is_at_infinity();
        let before_scale = (!was_at_infinity).then(|| {
            edited
                .base
                .image_table
                .placement_scale(&record.point.position)
        });
        let after = (decision.w != 0.0).then_some(&decision.position);
        if let Some(factor) = patch_frame_factor(&edited.base.image_table, before_scale, after) {
            for half in [&mut record.patch_u_halfvec, &mut record.patch_v_halfvec] {
                if let Some(v) = half.as_mut() {
                    for c in v.iter_mut() {
                        *c *= factor;
                    }
                }
            }
        }
        record.point.position = decision.position;
        record.point.w = decision.w;
        let to = next.replace_point(index, record)?;
        moves.push((index, to));
        report.moved += 1;
        report.crossed += usize::from(decision.crossed);
        shifts.extend(decision.shift);
    }
    // The middle of what the finite points travelled; see the field's own doc
    // for the population it is over.
    report.median_shift = if shifts.is_empty() {
        f64::NAN
    } else {
        median_in_place(&mut shifts)
    };
    Ok((next, PointMap::Replaced(moves), report))
}

/// The report every write-back starts from: what the solve was over, before any
/// point has been read back.
fn empty_report(gathered: &Gathered, estimates: &TriangulatedPoints) -> RetriangulateReport {
    RetriangulateReport {
        read: gathered.targets.len(),
        observations: gathered.obs_point.len(),
        held: gathered.held,
        moved: 0,
        crossed: 0,
        kept: 0,
        census: estimates.census,
        median_shift: f64::NAN,
    }
}

#[cfg(test)]
mod tests;
