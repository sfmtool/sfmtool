// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The two steps that estimate which way a track's patch faces and tilt it
//! there: [`fit_normal`], from the photographs' agreement over the patch, and
//! [`finite_difference_normal`], from where smaller pieces of the patch fit.
//!
//! `specs/core/bench/editable-track.md` § "Estimating the normal" is the
//! design. A [`fit`] moves the patch's centre and leaves its
//! normal as it was; these two do the opposite. Each estimates a normal, turns
//! the patch to it with [`tilt_patch`] -- so the turn
//! stops where that step's cap stops it, and every sighting keeps its in-plane
//! offset -- and ends as a fit does, by reading the result back and fusing the
//! consensus bitmap over the turned square. The centre does not move.

use nalgebra::{Point3, SymmetricEigen, Vector3};

use crate::patch::cloud::OrientedPatch;
use crate::patch::normal_refine::{refine_patch_normal, NormalRefineParams, ProjectedImage};
use crate::progress::{Cancelled, Progress};
use crate::progress_note;
use crate::reconstruction::edited::EditedReconstruction;

use super::evaluate::{check_views, evaluate, seed_of, EvaluateError, EvaluateReport};
use super::fit::{fit, fuse_bitmap_in_place, FitError, FitOptions};
use super::steps::{resize_patch, tilt_patch, translate_patch, TiltReport, TrackEditError};
use super::track::{EditableTrack, Stage, StageKind};

/// The smallest number of pieces [`FiniteDifferenceOptions::pieces`] takes: two
/// fitted centres are the fewest that fix a line.
pub const MIN_PIECES: usize = 2;

/// The largest number of pieces [`FiniteDifferenceOptions::pieces`] takes.
///
/// Each piece is one fit, and the pieces along an axis shrink as their count
/// grows, so past this a piece holds too little of the patch to localize.
pub const MAX_PIECES: usize = 8;

/// The largest overlap [`FiniteDifferenceOptions::overlap`] takes. At 1 every
/// piece would be the same piece.
pub const MAX_OVERLAP: f64 = 0.9;

/// The sine of the smallest angle two lines through the pieces' centres may
/// make and still be read as two directions. Below it the lines are one line,
/// which fixes only the tilt about its perpendicular.
const MIN_LINE_SINE: f64 = 0.17; // about 10 degrees

/// How [`fit_normal`] reads the photographs.
#[derive(Debug, Clone)]
pub struct FitNormalOptions {
    /// The photometric normal search. Its `min_views` defaults to two here,
    /// where the refinement kernel's own default is three, because the bench
    /// fits a track from two `in` sightings and a normal step should run on
    /// any track the fit runs on.
    pub refine: NormalRefineParams,
    /// The side of the square grid the consensus is scored on, in samples.
    pub resolution: u32,
    /// The reading the step ends with, and the fuse.
    pub fit: FitOptions,
}

impl Default for FitNormalOptions {
    fn default() -> Self {
        Self {
            refine: NormalRefineParams {
                min_views: 2,
                ..NormalRefineParams::default()
            },
            resolution: 24,
            fit: FitOptions::default(),
        }
    }
}

/// How [`finite_difference_normal`] cuts the patch into pieces.
#[derive(Debug, Clone)]
pub struct FiniteDifferenceOptions {
    /// How the pieces are laid out on the patch.
    pub layout: PieceLayout,
    /// How many pieces the patch is cut into along each of its two in-plane
    /// axes, from [`MIN_PIECES`] to [`MAX_PIECES`]: `pieces` in a row along
    /// each axis for [`PieceLayout::Cross`], `pieces × pieces` for
    /// [`PieceLayout::Grid`].
    pub pieces: usize,
    /// How much two neighbouring pieces overlap, as a fraction of a piece's
    /// side, from 0 to [`MAX_OVERLAP`].
    pub overlap: f64,
    /// The fit each piece is given, and the reading the step ends with.
    pub fit: FitOptions,
}

impl Default for FiniteDifferenceOptions {
    fn default() -> Self {
        Self {
            layout: PieceLayout::Cross,
            pieces: 2,
            overlap: 0.0,
            fit: FitOptions::default(),
        }
    }
}

/// Where [`finite_difference_normal`] places its pieces.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PieceLayout {
    /// A row of pieces through the centre along each in-plane axis, each row
    /// giving a line in the surface; the normal is the cross product of the
    /// two lines.
    Cross,
    /// A grid of pieces tiling the whole patch; the normal is that of the
    /// least-squares plane through every fitted centre.
    Grid,
}

impl FiniteDifferenceOptions {
    /// Whether the piece count and the overlap are ones the step takes.
    ///
    /// Published so a caller can refuse settings before it decodes the
    /// photographs; [`finite_difference_normal`] asks the same question.
    pub fn check(&self) -> Result<(), NormalError> {
        if !(MIN_PIECES..=MAX_PIECES).contains(&self.pieces) {
            return Err(NormalError::BadPieces(self.pieces));
        }
        if !(0.0..=MAX_OVERLAP).contains(&self.overlap) {
            return Err(NormalError::BadOverlap(self.overlap));
        }
        Ok(())
    }
}

/// Why a normal step was refused. Every variant names what did not hold, in
/// a sentence a button can show.
#[derive(Debug, Clone, PartialEq)]
pub enum NormalError {
    /// The track is at the cluster stage, which has no patch in the world to
    /// turn.
    ClusterStage,
    /// The track carries no patch.
    NoFrame,
    /// The track is at infinity, where the patch's normal is its own bearing.
    AtInfinity,
    /// Fewer than two observations are `in`, so there is no consensus to read
    /// a normal from.
    TooFewObservations(usize),
    /// [`FiniteDifferenceOptions::pieces`] is outside
    /// [`MIN_PIECES`]..=[`MAX_PIECES`].
    BadPieces(usize),
    /// [`FiniteDifferenceOptions::overlap`] is outside 0..=[`MAX_OVERLAP`].
    BadOverlap(f64),
    /// The photometric search scored no normal: fewer views than it needs were
    /// valid at the patch.
    NotScored {
        /// How many views its validity gates kept.
        views: u32,
    },
    /// Fewer than two pieces fitted along either axis, so no line through
    /// their centres exists.
    TooFewPieces {
        /// How many pieces fitted along the patch's `u` axis.
        u: usize,
        /// How many along its `v` axis.
        v: usize,
    },
    /// Fewer than two pieces of the grid fitted, so no line or plane through
    /// their centres exists.
    TooFewGridPieces {
        /// How many pieces fitted.
        fitted: usize,
        /// How many the grid holds.
        of: usize,
    },
    /// The turn to the estimated normal was refused.
    Tilt(TrackEditError),
    /// The reading the step ends with failed.
    Evaluate(EvaluateError),
    /// The caller asked the step to stop.
    Cancelled,
}

impl std::fmt::Display for NormalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NormalError::ClusterStage => write!(
                f,
                "the track is at the cluster stage, which has no patch to turn; upgrade it \
                 to the track stage first"
            ),
            NormalError::NoFrame => write!(f, "the track carries no patch to turn"),
            NormalError::AtInfinity => write!(
                f,
                "the track is at infinity, where the patch faces along its own bearing"
            ),
            NormalError::TooFewObservations(n) => {
                write!(f, "{n} observations are in, and a normal needs two or more")
            }
            NormalError::BadPieces(n) => write!(
                f,
                "{n} pieces per axis were asked for, and the patch is cut into \
                 {MIN_PIECES} to {MAX_PIECES} per axis"
            ),
            NormalError::BadOverlap(o) => write!(
                f,
                "an overlap of {:.0}% was asked for, and it runs from 0% to {:.0}%",
                o * 100.0,
                MAX_OVERLAP * 100.0
            ),
            NormalError::NotScored { views } => write!(
                f,
                "the photographs scored no normal: {views} views were valid at the patch"
            ),
            NormalError::TooFewPieces { u, v } => write!(
                f,
                "{u} pieces fitted along u and {v} along v, and a line needs two along one \
                 axis"
            ),
            NormalError::TooFewGridPieces { fitted, of } => write!(
                f,
                "{fitted} of the grid's {of} pieces fitted, and a plane needs more"
            ),
            NormalError::Tilt(e) => write!(f, "{e}"),
            NormalError::Evaluate(e) => write!(f, "{e}"),
            NormalError::Cancelled => write!(f, "the step was cancelled"),
        }
    }
}

impl std::error::Error for NormalError {}

impl From<Cancelled> for NormalError {
    fn from(_: Cancelled) -> Self {
        NormalError::Cancelled
    }
}

impl From<EvaluateError> for NormalError {
    fn from(e: EvaluateError) -> Self {
        match e {
            EvaluateError::Cancelled => NormalError::Cancelled,
            e => NormalError::Evaluate(e),
        }
    }
}

impl From<TrackEditError> for NormalError {
    fn from(e: TrackEditError) -> Self {
        match e {
            TrackEditError::AtInfinity => NormalError::AtInfinity,
            e => NormalError::Tilt(e),
        }
    }
}

/// Where the normal a step turned the patch to came from.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NormalEstimate {
    /// [`fit_normal`]'s photometric search.
    Photometric {
        /// The consensus ZNCC at the normal the patch had.
        before: f64,
        /// The consensus ZNCC at the normal it found.
        after: f64,
        /// How many views the search scored.
        views: u32,
    },
    /// [`finite_difference_normal`]'s pieces, in a [`PieceLayout::Cross`].
    FiniteDifference {
        /// How many pieces were cut along each axis.
        pieces: usize,
        /// How many pieces fitted along the patch's `u` axis.
        u: usize,
        /// How many along its `v` axis.
        v: usize,
        /// Whether the two lines through them fixed the whole normal. `false`
        /// when only one line was usable, which fixes the tilt about its
        /// perpendicular and leaves the tilt about the line as it was.
        both_axes: bool,
    },
    /// [`finite_difference_normal`]'s pieces, in a [`PieceLayout::Grid`].
    GridPlane {
        /// How many pieces were cut along each axis, `pieces × pieces` in all.
        pieces: usize,
        /// How many of them fitted.
        fitted: usize,
        /// Whether the fitted centres spread in two directions and so fixed
        /// the whole normal. `false` when they lay along a line, which fixes
        /// only the turn about its perpendicular.
        both_axes: bool,
        /// The rms distance of the fitted centres from their plane, in the
        /// patch's half-lengths: how flat the surface under the patch is.
        /// `NaN` when the centres fixed no plane.
        off_plane: f64,
    },
}

/// What one normal step did.
#[derive(Debug, Clone, PartialEq)]
pub struct NormalReport {
    /// Where the normal came from.
    pub estimate: NormalEstimate,
    /// The turn: the normal asked for, the one reached, and the observation
    /// whose cap stopped it short, if one did.
    pub tilt: TiltReport,
    /// The reading of the turned track.
    pub evaluate: EvaluateReport,
}

impl std::fmt::Display for NormalReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.estimate {
            NormalEstimate::Photometric {
                before,
                after,
                views,
            } => write!(f, "ZNCC {before:.3} \u{23f5} {after:.3} over {views} views")?,
            NormalEstimate::FiniteDifference {
                pieces,
                u,
                v,
                both_axes,
            } => {
                write!(
                    f,
                    "{pieces} pieces along each axis, {u} fitted along u and {v} along v"
                )?;
                if !both_axes {
                    write!(f, ", one axis fixed")?;
                }
            }
            NormalEstimate::GridPlane {
                pieces,
                fitted,
                both_axes,
                off_plane,
            } => {
                write!(
                    f,
                    "{pieces}x{pieces} pieces, {fitted} of {} fitted",
                    pieces * pieces
                )?;
                if both_axes {
                    write!(f, ", rms {off_plane:.3} half-lengths off their plane")?;
                } else {
                    write!(f, ", one axis fixed")?;
                }
            }
        }
        write!(f, ", turned {:.1}\u{b0}", self.tilt.degrees)?;
        if let Some(stop) = self.tilt.stopped {
            write!(f, " (stopped by image {})", stop.image)?;
        }
        write!(f, ", {}", self.evaluate)
    }
}

/// Whether a normal step can run on `track`, judged on the track alone.
///
/// The half of [`fit_normal`]'s and [`finite_difference_normal`]'s validation
/// that reads no photograph, for a caller that would decode the photographs
/// first to refuse in front of that work.
///
/// # Example
///
/// ```no_run
/// # use sfmtool_core::bench::{normal_preconditions, EditableTrack};
/// # fn run(track: &EditableTrack) -> Result<(), Box<dyn std::error::Error>> {
/// normal_preconditions(track)?;   // refuse here, before a photograph is read
/// # Ok(())
/// # }
/// ```
pub fn normal_preconditions(track: &EditableTrack) -> Result<(), NormalError> {
    let Stage::Track(payload) = &track.stage else {
        return Err(NormalError::ClusterStage);
    };
    let Some(frame) = &payload.placement else {
        return Err(NormalError::NoFrame);
    };
    if payload.at_infinity || frame.w == 0.0 {
        return Err(NormalError::AtInfinity);
    }
    let ins = track.in_observations().len();
    if ins < 2 {
        return Err(NormalError::TooFewObservations(ins));
    }
    Ok(())
}

/// Turn the track's patch to the normal the photographs agree on most.
///
/// The search is the photometric normal refinement
/// (`specs/core/patch/patch-normal-refinement.md`) over the `in` sightings,
/// each view's tile anchored at the sighting's own keypoint. It is seeded from
/// the normal the patch has and from the mean viewing direction and searches
/// [`NormalRefineParams::angular_range_deg`] around each, so a normal further
/// away than that is reached by pressing it again. The patch is then turned
/// with [`tilt_patch`], read back and fused, and its centre does not move.
///
/// `images` is one [`ProjectedImage`] per image of `edited`, as for every
/// photometric step of the bench.
///
/// # Example
///
/// ```no_run
/// # use sfmtool_core::bench::{fit_normal, FitNormalOptions};
/// # use sfmtool_core::progress::Progress;
/// # fn run(
/// #     track: &sfmtool_core::bench::EditableTrack,
/// #     edited: &sfmtool_core::EditedReconstruction,
/// #     images: &[sfmtool_core::patch::normal_refine::ProjectedImage<'_>],
/// # ) -> Result<(), Box<dyn std::error::Error>> {
/// let (turned, report) =
///     fit_normal(track, edited, images, &FitNormalOptions::default(), &Progress::none())?;
/// println!("{report}");   // "ZNCC 0.712 ⏵ 0.845 over 6 views, turned 14.2°, …"
/// # let _ = turned;
/// # Ok(())
/// # }
/// ```
pub fn fit_normal(
    track: &EditableTrack,
    edited: &EditedReconstruction,
    images: &[ProjectedImage<'_>],
    options: &FitNormalOptions,
    progress: &Progress<'_>,
) -> Result<(EditableTrack, NormalReport), NormalError> {
    check_views(edited, images)?;
    normal_preconditions(track)?;
    let frame = frame_of(track);

    let mut views: Vec<ProjectedImage<'_>> = Vec::new();
    let mut keypoints: Vec<Option<[f64; 2]>> = Vec::new();
    let mut seen = std::collections::BTreeSet::new();
    for &i in &track.in_observations() {
        let observation = &track.observations[i];
        let Some(view) = images.get(observation.image as usize) else {
            continue;
        };
        if seen.insert(observation.image) {
            views.push(*view);
            keypoints.push(seed_of(observation));
        }
    }

    progress.check_cancel()?;
    let result = {
        let mut phase = progress.phase("normal");
        let result = refine_patch_normal(
            &frame,
            &views,
            options.resolution,
            &options.refine,
            Some(&keypoints),
        );
        progress_note!(phase, "{} views", views.len());
        result
    };
    if !result.photoconsistency.is_finite() {
        return Err(NormalError::NotScored {
            views: result.valid_view_count,
        });
    }
    let estimate = NormalEstimate::Photometric {
        before: result.init_photoconsistency,
        after: result.photoconsistency,
        views: result.valid_view_count,
    };
    turn_and_read(
        track,
        edited,
        images,
        result.patch.normal(),
        estimate,
        &options.fit,
        progress,
    )
}

/// Turn the track's patch to the plane its pieces fit on.
///
/// The patch is cut, along each of its two in-plane axes, into
/// [`FiniteDifferenceOptions::pieces`] square pieces that together span its
/// side, neighbours overlapping by [`FiniteDifferenceOptions::overlap`] of a
/// piece's side. Each piece is a copy of the track resized and slid along the
/// plane, and is given a [`fit`], which moves it along the sightings' rays to
/// where the photographs put it. The fitted centres along one axis lie on the
/// surface, so the line through them (the direction they spread in most) lies
/// in it; the two lines fix the normal. A piece whose fit fails, or which the
/// fit puts at infinity, is left out.
///
/// Where only one axis has two fitted pieces, or the two lines are nearly
/// parallel, the one line fixes the turn about its perpendicular, and the tilt
/// about the line itself is left as the patch had it.
///
/// A finite difference of the surface's depth across the patch, taken from
/// where the photographs place each piece, rather than from how well one plane
/// matches them all.
///
/// # Example
///
/// ```no_run
/// # use sfmtool_core::bench::{finite_difference_normal, FiniteDifferenceOptions};
/// # use sfmtool_core::progress::Progress;
/// # fn run(
/// #     track: &sfmtool_core::bench::EditableTrack,
/// #     edited: &sfmtool_core::EditedReconstruction,
/// #     images: &[sfmtool_core::patch::normal_refine::ProjectedImage<'_>],
/// # ) -> Result<(), Box<dyn std::error::Error>> {
/// let options = FiniteDifferenceOptions { pieces: 3, overlap: 0.25, ..Default::default() };
/// let (turned, report) =
///     finite_difference_normal(track, edited, images, &options, &Progress::none())?;
/// println!("{report}");   // "3 pieces fitted along u, 3 along v, turned 8.0°, …"
/// # let _ = turned;
/// # Ok(())
/// # }
/// ```
pub fn finite_difference_normal(
    track: &EditableTrack,
    edited: &EditedReconstruction,
    images: &[ProjectedImage<'_>],
    options: &FiniteDifferenceOptions,
    progress: &Progress<'_>,
) -> Result<(EditableTrack, NormalReport), NormalError> {
    check_views(edited, images)?;
    normal_preconditions(track)?;
    options.check()?;
    let frame = frame_of(track);
    let was = frame.normal();
    let (piece_half, offsets) = piece_layout(frame.half_extent[0], options.pieces, options.overlap);
    let pieces = Pieces {
        track,
        edited,
        images,
        options,
        progress,
        was,
        half: piece_half,
        offsets: &offsets,
    };
    let (normal, estimate) = match options.layout {
        PieceLayout::Cross => cross_normal(&pieces)?,
        PieceLayout::Grid => grid_normal(&pieces, frame.half_extent[0])?,
    };
    // A line or a plane has no sign: keep the face the patch showed.
    let normal = if normal.dot(&was) < 0.0 {
        -normal
    } else {
        normal
    };
    turn_and_read(
        track,
        edited,
        images,
        normal,
        estimate,
        &options.fit,
        progress,
    )
}

/// What every piece of one [`finite_difference_normal`] shares.
struct Pieces<'a, 'v> {
    track: &'a EditableTrack,
    edited: &'a EditedReconstruction,
    images: &'a [ProjectedImage<'v>],
    options: &'a FiniteDifferenceOptions,
    progress: &'a Progress<'a>,
    /// The normal the patch had.
    was: Vector3<f64>,
    /// A piece's half-length.
    half: f64,
    /// Each piece's offset from the centre along one axis.
    offsets: &'a [f64],
}

impl Pieces<'_, '_> {
    /// Fit the piece slid by `by` on the patch's own axes, and give back its
    /// centre, or `None` where it did not fit.
    fn fit(&self, by: Vector3<f64>) -> Result<Option<Point3<f64>>, NormalError> {
        self.progress.check_cancel()?;
        fit_piece(
            self.track,
            self.edited,
            self.images,
            self.half,
            by,
            self.options,
            self.progress,
        )
    }
}

/// [`PieceLayout::Cross`]: a row of pieces along each axis, a line through
/// each row's fitted centres, and the normal across the two lines.
fn cross_normal(pieces: &Pieces<'_, '_>) -> Result<(Vector3<f64>, NormalEstimate), NormalError> {
    let was = pieces.was;
    let mut lines: Vec<Vector3<f64>> = Vec::with_capacity(2);
    let mut fitted = [0usize; 2];
    for (axis, slot) in fitted.iter_mut().enumerate() {
        let mut centres: Vec<Point3<f64>> = Vec::with_capacity(pieces.offsets.len());
        for &offset in pieces.offsets {
            let mut by = Vector3::zeros();
            by[axis] = offset;
            if let Some(centre) = pieces.fit(by)? {
                centres.push(centre);
            }
        }
        *slot = centres.len();
        if let Some(line) = principal_direction(&centres) {
            lines.push(line);
        }
    }

    let (normal, both_axes) = match lines.as_slice() {
        [a, b] if a.cross(b).norm() >= MIN_LINE_SINE => (a.cross(b).normalize(), true),
        [a, ..] => (perpendicular_part(was, *a).unwrap_or(was), false),
        [] => {
            return Err(NormalError::TooFewPieces {
                u: fitted[0],
                v: fitted[1],
            })
        }
    };
    Ok((
        normal,
        NormalEstimate::FiniteDifference {
            pieces: pieces.options.pieces,
            u: fitted[0],
            v: fitted[1],
            both_axes,
        },
    ))
}

/// [`PieceLayout::Grid`]: a grid of pieces tiling the patch, and the normal of
/// the least-squares plane through their fitted centres. `half` is the whole
/// patch's half-length, which the report's flatness is stated in.
///
/// The plane's normal is the direction the centres spread in least. Where the
/// middle spread is under [`MIN_LINE_SINE`] of the largest, the centres lie
/// along a line, which fixes only the turn about its perpendicular, as a
/// single row of the cross does.
fn grid_normal(
    pieces: &Pieces<'_, '_>,
    half: f64,
) -> Result<(Vector3<f64>, NormalEstimate), NormalError> {
    let offsets = pieces.offsets;
    let mut centres: Vec<Point3<f64>> = Vec::with_capacity(offsets.len() * offsets.len());
    for &along_v in offsets {
        for &along_u in offsets {
            if let Some(centre) = pieces.fit(Vector3::new(along_u, along_v, 0.0))? {
                centres.push(centre);
            }
        }
    }
    let too_few = NormalError::TooFewGridPieces {
        fitted: centres.len(),
        of: offsets.len() * offsets.len(),
    };
    let Some(spread) = spread_of(&centres) else {
        return Err(too_few);
    };
    let [least, middle, most] = spread.values;
    if most <= 0.0 {
        return Err(too_few);
    }
    // The spreads are variances, so the angle test reads their square roots.
    let both_axes = (middle / most).max(0.0).sqrt() >= MIN_LINE_SINE;
    let (normal, off_plane) = if both_axes {
        let off = if half > 0.0 {
            least.max(0.0).sqrt() / half
        } else {
            f64::NAN
        };
        (spread.directions[0], off)
    } else {
        let line = spread.directions[2];
        (
            perpendicular_part(pieces.was, line).unwrap_or(pieces.was),
            f64::NAN,
        )
    };
    Ok((
        normal,
        NormalEstimate::GridPlane {
            pieces: pieces.options.pieces,
            fitted: centres.len(),
            both_axes,
            off_plane,
        },
    ))
}

/// The square patch the track carries, which [`normal_preconditions`] has
/// checked is there.
fn frame_of(track: &EditableTrack) -> OrientedPatch {
    track
        .track()
        .and_then(|payload| payload.placement.as_ref())
        .expect("normal_preconditions checked the patch is there")
        .squared()
}

/// A piece's half-length, and the offset of each piece's centre from the
/// patch's along one axis, for `pieces` pieces overlapping by `overlap` of a
/// side and together spanning the patch's side `2 half`.
///
/// With side `s`, the pieces start `s (1 - overlap)` apart, so `n` of them span
/// `s + (n - 1) s (1 - overlap)`, which is `2 half` when
/// `s = 2 half / (n - (n - 1) overlap)`. Two pieces with no overlap are the
/// halves of the patch, a half-length `half / 2` each at `±half / 2`.
fn piece_layout(half: f64, pieces: usize, overlap: f64) -> (f64, Vec<f64>) {
    let n = pieces as f64;
    let side = 2.0 * half / (n - (n - 1.0) * overlap);
    let step = side * (1.0 - overlap);
    let first = -half + side / 2.0;
    let offsets = (0..pieces).map(|k| first + k as f64 * step).collect();
    (side / 2.0, offsets)
}

/// Fit one piece: the track resized to `half` about its centre, slid by `by` on
/// its own axes, and fitted. Its centre, or `None` where the fit refused it or
/// put it at infinity.
fn fit_piece(
    track: &EditableTrack,
    edited: &EditedReconstruction,
    images: &[ProjectedImage<'_>],
    half: f64,
    by: Vector3<f64>,
    options: &FiniteDifferenceOptions,
    progress: &Progress<'_>,
) -> Result<Option<Point3<f64>>, NormalError> {
    let Ok((piece, _)) = resize_patch(track, edited, half, None) else {
        return Ok(None);
    };
    let Ok((piece, _)) = translate_patch(&piece, edited, by) else {
        return Ok(None);
    };
    match fit(&piece, edited, images, &options.fit, progress) {
        Err(FitError::Cancelled) => Err(NormalError::Cancelled),
        Err(_) => Ok(None),
        Ok((_, report)) => Ok(report
            .classification
            .filter(|call| !call.at_infinity)
            .and(report.position)
            .filter(|p| p.coords.iter().all(|c| c.is_finite()))),
    }
}

/// How a set of points spreads about its mean: the variance along each
/// principal direction, least first, with the unit directions in the same
/// order.
struct Spread {
    values: [f64; 3],
    directions: [Vector3<f64>; 3],
}

/// The spread of `points`, or `None` for fewer than two points or a result
/// that is not finite.
fn spread_of(points: &[Point3<f64>]) -> Option<Spread> {
    if points.len() < 2 {
        return None;
    }
    let n = points.len() as f64;
    let mean = points
        .iter()
        .fold(Vector3::zeros(), |sum, p| sum + p.coords)
        / n;
    let scatter = points.iter().fold(nalgebra::Matrix3::zeros(), |sum, p| {
        let d = p.coords - mean;
        sum + d * d.transpose()
    }) / n;
    let eigen = SymmetricEigen::new(scatter);
    let mut order = [0usize, 1, 2];
    order.sort_by(|&a, &b| eigen.eigenvalues[a].total_cmp(&eigen.eigenvalues[b]));
    let values = order.map(|k| eigen.eigenvalues[k]);
    let directions = order.map(|k| eigen.eigenvectors.column(k).into_owned());
    let finite = values.iter().all(|v| v.is_finite())
        && directions
            .iter()
            .flat_map(|d| d.iter())
            .all(|c| c.is_finite());
    finite.then(|| Spread {
        values,
        directions: directions.map(|d| d.normalize()),
    })
}

/// The direction `points` spread in most, or `None` for fewer than two points
/// or points that do not spread.
fn principal_direction(points: &[Point3<f64>]) -> Option<Vector3<f64>> {
    let spread = spread_of(points)?;
    (spread.values[2] > 0.0).then_some(spread.directions[2])
}

/// `normal` with its component along the unit `line` removed, normalized: the
/// normal nearest it that is perpendicular to the line. `None` where the two
/// are parallel.
fn perpendicular_part(normal: Vector3<f64>, line: Vector3<f64>) -> Option<Vector3<f64>> {
    let across = normal - line * normal.dot(&line);
    (across.norm() > 1e-9).then(|| across.normalize())
}

/// Turn the track to `normal` with [`tilt_patch`], read the result back and
/// fuse its bitmap, as a fit ends.
fn turn_and_read(
    track: &EditableTrack,
    edited: &EditedReconstruction,
    images: &[ProjectedImage<'_>],
    normal: Vector3<f64>,
    estimate: NormalEstimate,
    options: &FitOptions,
    progress: &Progress<'_>,
) -> Result<(EditableTrack, NormalReport), NormalError> {
    progress.check_cancel()?;
    let (turned, tilt) = tilt_patch(track, edited, normal)?;
    let (read, evaluate) = evaluate(&turned, edited, images, &options.evaluate, progress)?;
    debug_assert_eq!(read.stage_kind(), StageKind::Track);
    let fused = fuse_bitmap_in_place(&read, edited, images, options);
    Ok((
        fused,
        NormalReport {
            estimate,
            tilt,
            evaluate,
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_pieces_with_no_overlap_are_the_halves() {
        let (half, offsets) = piece_layout(2.0, 2, 0.0);
        assert!((half - 1.0).abs() < 1e-12);
        assert_eq!(offsets.len(), 2);
        assert!((offsets[0] + 1.0).abs() < 1e-12);
        assert!((offsets[1] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn overlapping_pieces_span_the_patch() {
        for pieces in MIN_PIECES..=MAX_PIECES {
            for overlap in [0.0, 0.25, 0.5, MAX_OVERLAP] {
                let (half, offsets) = piece_layout(3.0, pieces, overlap);
                let first = offsets[0] - half;
                let last = offsets[pieces - 1] + half;
                assert!((first + 3.0).abs() < 1e-9, "{pieces} {overlap}: {first}");
                assert!((last - 3.0).abs() < 1e-9, "{pieces} {overlap}: {last}");
                if pieces > 1 {
                    let gap = offsets[1] - offsets[0];
                    assert!((gap - 2.0 * half * (1.0 - overlap)).abs() < 1e-9);
                }
            }
        }
    }

    #[test]
    fn a_line_through_points_is_their_spread() {
        let points = [
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.1),
            Point3::new(2.0, 0.0, 0.2),
        ];
        let line = principal_direction(&points).expect("the points spread");
        let expected = Vector3::new(1.0, 0.0, 0.1).normalize();
        assert!(line.dot(&expected).abs() > 1.0 - 1e-9);
        assert!(principal_direction(&points[..1]).is_none());
        assert!(principal_direction(&[points[0], points[0]]).is_none());
    }

    #[test]
    fn one_line_keeps_the_tilt_about_itself() {
        let was = Vector3::new(0.0, 0.3, 1.0).normalize();
        let line = Vector3::new(1.0, 0.0, 0.5).normalize();
        let n = perpendicular_part(was, line).expect("not parallel");
        assert!(n.dot(&line).abs() < 1e-12);
        // Nothing turned about the line: the part of the old normal across it
        // is where the new one points.
        assert!(n.dot(&was) > 0.9);
    }
}
