// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The evaluation: reading the track as it stands, at whichever stage it is in.
//!
//! `specs/core/bench/editable-track.md` is the design. [`evaluate`] is a pure
//! function from a track, the reconstruction it belongs to and one decoded view
//! per image to the same track with its measurement slots filled. It moves
//! nothing: the position, the frame, the bitmap, the keypoints and every
//! verdict come back exactly as they went in, and what changes is the report
//! each observation carries about itself. The step that *changes* the track is
//! [`fit`](super::fit::fit), which ends by calling this so that a fit and a
//! reading of its result can never disagree about a number.
//!
//! **Nothing is dropped.** The kernels are run with their per-view gates off
//! and their consensus-basis cap lifted ([`EvaluateOptions::default`]), because
//! a gate is a decision and this step makes none: an observation the pipeline
//! would have thrown away is exactly the one the person is looking at. An
//! observation that genuinely cannot be read comes back with a
//! [`Unmeasured`] reason instead of a blank row.
//!
//! At the **cluster stage** the reading is the `.matches` refinement kernel over
//! an in-memory cluster, whose `member_status` is its own account of every
//! member; at the **track stage** it is one round of
//! [`try_localize_patch_keypoints`] over the observations where they sit, which
//! scores each one against the leave-one-out consensus of the others and finds
//! its correlation peak without moving it there.

use std::collections::HashMap;

use nalgebra::{Point3, Vector3};
use ndarray::{Array2, Array3};

use crate::camera::image::ImageU8Pyramid;

use crate::patch::cloud::OrientedPatch;
use crate::patch::cluster_refine::{
    refine_cluster_patches_borrowed, sample_member_grid, ClusterRefineParams, FeatureGeometry,
    MemberStatus, REFERENCE_UNREFINABLE,
};
use crate::patch::keypoint_localize::{
    keypoint_grid_offset, project_unclipped, try_localize_patch_keypoints, view_cache_bytes,
    KeypointLocalizeParams, LocalizeError,
};
use crate::patch::member_coherence::{member_zncc_matrix_reporting, MemberCoherenceParams};
use crate::patch::normal_refine::ProjectedImage;
use crate::patch::reference_view::{
    cell_agreement, choose_reference_view, finite_middle, render_view_tile, ReferenceReadings,
    ViewTile,
};
use crate::patch::self_similarity::{
    zncc_self_similarity_parts, PatchTile, SelfSimilarityEllipse, SelfSimilarityEllipseUnits,
    SelfSimilarityParams,
};
use crate::progress::{Cancelled, Progress};
use crate::progress_note;
use crate::reconstruction::edited::EditedReconstruction;
use crate::SfmrReconstruction;

use super::steps::apply_thresholds;
use super::track::{
    ClusterPayload, ClusterTemplate, EditableTrack, Observation, RepaintMark, Stage, StageKind,
    TrackPayload, Unmeasured, Verdict,
};

/// The localizer with every per-view gate off and the consensus-basis cap
/// lifted: what both an evaluation and a fit run.
///
/// Of the four gates the kernel applies per view,
/// [`min_absolute_zncc`](KeypointLocalizeParams::min_absolute_zncc),
/// [`min_relative_zncc`](KeypointLocalizeParams::min_relative_zncc) and
/// [`max_member_zncc_self_similarity_radius`](KeypointLocalizeParams::max_member_zncc_self_similarity_radius)
/// are off, and [`max_shift_px`](KeypointLocalizeParams::max_shift_px) is a
/// large finite bar so that the infinite shift the kernel reports for a
/// keypoint that left the photograph still lands. `basis_max_views` is `0`.
/// Why the bench drops no sighting in the kernel is in
/// `specs/core/bench/editable-track.md` § "Why it is shaped this way".
pub fn open_localizer() -> KeypointLocalizeParams {
    KeypointLocalizeParams {
        // Disabled exactly: each of the three reads "0 or non-finite is off".
        max_member_zncc_self_similarity_radius: 0.0,
        min_absolute_zncc: 0.0,
        min_relative_zncc: 0.0,
        max_shift_px: f64::MAX,
        // Every view congeals; nothing registers against a finished template.
        basis_max_views: 0,
        ..KeypointLocalizeParams::default()
    }
}

/// What the kernels an evaluation runs are allowed to do.
///
/// The track's [`Thresholds`](super::track::Thresholds) are **not** here on
/// purpose: a bar the person moved paints verdicts, and it must not change what
/// the numbers under the painting are.
#[derive(Debug, Clone)]
pub struct EvaluateOptions {
    /// The cluster stage's refinement kernel.
    ///
    /// Its
    /// [`radius`](crate::patch::cluster_refine::ClusterRefineParams::radius) is
    /// the one field an evaluation does **not** take from here: the template's
    /// half-width is
    /// [`ClusterPayload::radius`](super::track::ClusterPayload::radius),
    /// because it is what every one of the track's shapes was written against
    /// and a round run at another radius would be measuring a different square
    /// from the one the person put on the bench.
    pub cluster: ClusterRefineParams,
    /// The track stage's localizer. Defaults to [`open_localizer`] with one
    /// round: a reading registers nothing, so the congealing loop that would
    /// walk every view towards a shared optimum is run once, over the
    /// observations where they already sit.
    ///
    /// Its [`resolution`](KeypointLocalizeParams::resolution) applies only to
    /// a reconstruction that stores no patch bitmaps. Otherwise the track
    /// stage reads at the reconstruction's own patch resolution
    /// ([`Self::patch_resolution`]), so every patch-grid px it reports is a
    /// px of the patch the reconstruction holds.
    pub localize: KeypointLocalizeParams,
    /// How far from the projection a seed may sit and still be read, in
    /// **patch-grid px**. An observation past it is left out of the round
    /// carrying [`Unmeasured::SeedTooFar`]. Defaults to
    /// [`DEFAULT_MAX_SEED_OFFSET_PX`]; why the bound exists and why it is 64 is
    /// in `specs/core/bench/editable-track.md` § "Evaluating".
    pub max_seed_offset_px: f64,
    /// The most memory one round's per-view tiles may take together, in bytes.
    ///
    /// [`Self::max_seed_offset_px`] bounds one observation and this bounds the
    /// round: a track with enough views at the bound would still add up to more
    /// than the machine has. A round past it is refused with
    /// [`EvaluateError::TooLarge`] **before** anything is allocated, which is a
    /// sentence the person reads rather than an allocator abort that takes the
    /// window with it. The default is 256 MiB, which is a couple of hundred
    /// views at the offset bound and thousands at the default search radius.
    pub max_cache_bytes: usize,
}

/// [`EvaluateOptions::max_seed_offset_px`]'s default: 64 patch-grid px.
pub const DEFAULT_MAX_SEED_OFFSET_PX: f64 = 64.0;

/// [`EvaluateOptions::max_cache_bytes`]'s default: 256 MiB.
pub const DEFAULT_MAX_CACHE_BYTES: usize = 256 << 20;

impl Default for EvaluateOptions {
    fn default() -> Self {
        Self {
            // The bench judges a sighting's tile by its self-similarity radius
            // in its painting (`Thresholds::max_zncc_self_similarity_radius`),
            // so the refinement's own member gate on the same radius is off:
            // a gate is a decision, and the reading makes none.
            cluster: ClusterRefineParams {
                max_member_zncc_self_similarity_radius: 0.0,
                ..ClusterRefineParams::default()
            },
            localize: KeypointLocalizeParams {
                max_iters: 1,
                ..open_localizer()
            },
            max_seed_offset_px: DEFAULT_MAX_SEED_OFFSET_PX,
            max_cache_bytes: DEFAULT_MAX_CACHE_BYTES,
        }
    }
}

impl EvaluateOptions {
    /// The patch-grid resolution `R`, in grid px a side, that a track-stage
    /// reading of a track on `recon` is stated in: the edge of `recon`'s patch
    /// bitmaps where it stores them ([`stored_patch_resolution`]), else
    /// [`Self::localize`]'s resolution.
    ///
    /// Every patch-grid px the track stage reports is one `R`-th of the
    /// patch's side: the shift, the self-similarity radius and its ellipse. A
    /// caller that states a patch's zoom in grid px per photograph pixel uses
    /// the same `R`, so the zoom and the ellipse are in one unit.
    ///
    /// ```no_run
    /// # use sfmtool_core::bench::EvaluateOptions;
    /// # fn run(recon: &sfmtool_core::SfmrReconstruction) {
    /// let r = EvaluateOptions::default().patch_resolution(recon); // 24 unless the file says otherwise
    /// # let _ = r;
    /// # }
    /// ```
    pub fn patch_resolution(&self, recon: &SfmrReconstruction) -> u32 {
        stored_patch_resolution(recon)
            .unwrap_or(self.localize.resolution)
            .max(2)
    }

    /// These options with the track stage's localizer at `resolution`.
    pub(super) fn at_resolution(&self, resolution: u32) -> Self {
        let mut options = self.clone();
        options.localize.resolution = resolution;
        options
    }
}

/// The edge `R` of `recon`'s `(P, R, R, 4)` patch bitmaps, which an `.sfmr`
/// declares as `patch_bitmap_resolution`, or `None` where it stores none.
///
/// A column the viewer rendered for display because the file had none counts
/// as well, as it does for the fit's fuse: the bench reads it as it would the
/// file's own.
pub fn stored_patch_resolution(recon: &SfmrReconstruction) -> Option<u32> {
    let bitmaps = recon.point_set.patch_bitmaps_y_x_rgba.as_deref()?;
    u32::try_from(bitmaps.shape()[1]).ok().filter(|&r| r > 0)
}

/// Why an evaluation was refused. Every variant names what did not hold,
/// because the caller is a button that has to say so in one sentence.
///
/// There is no "too few observations" among them: a reading of a track with one
/// sighting in it is a reading that reports one sighting as having nothing to
/// correlate against, which is a measurement and not a refusal.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EvaluateError {
    /// Fewer decoded views were supplied than the reconstruction has images.
    ViewsMissing {
        /// How many were supplied.
        got: usize,
        /// How many the reconstruction holds.
        expected: usize,
    },
    /// An observation names an image no decoded, posed view was supplied for.
    NoView {
        /// The image it named.
        image: u32,
    },
    /// The track carries no patch, so there is nothing any view can be read
    /// against.
    NoFrame,
    /// One round's per-view tiles would take more memory than
    /// [`EvaluateOptions::max_cache_bytes`] allows.
    ///
    /// Refused in front of the allocation rather than attempted: the tiles are
    /// sized by the widened search window and an allocation the global
    /// allocator cannot make aborts the process.
    TooLarge {
        /// What the round's tiles would have taken, in bytes.
        bytes: usize,
        /// The budget it passed.
        budget: usize,
    },
    /// A buffer the localizer needed could not be allocated, at a size the
    /// budget admitted.
    OutOfMemory {
        /// What the allocator refused, in bytes.
        bytes: usize,
    },
    /// The caller asked the evaluation to stop.
    Cancelled,
}

impl std::fmt::Display for EvaluateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EvaluateError::ViewsMissing { got, expected } => write!(
                f,
                "{got} decoded views were supplied and the reconstruction has {expected} images"
            ),
            EvaluateError::NoView { image } => write!(
                f,
                "image {image} has no decoded view with a pose in this evaluation"
            ),
            EvaluateError::NoFrame => write!(
                f,
                "the track carries no patch frame to read against, as a point put on \
                 the bench from a sift_files reconstruction has none; convert the \
                 reconstruction to embedded patches and put the point on the bench again"
            ),
            EvaluateError::TooLarge { bytes, budget } => write!(
                f,
                "this round's tiles would take {} and the budget is {}; \
                 narrow the search or turn out the observations furthest from \
                 the projection",
                shown_bytes(*bytes),
                shown_bytes(*budget)
            ),
            EvaluateError::OutOfMemory { bytes } => {
                write!(f, "{bytes} bytes could not be allocated for the reading")
            }
            EvaluateError::Cancelled => write!(f, "the reading was cancelled"),
        }
    }
}

impl std::error::Error for EvaluateError {}

/// A byte count as a refusal says it: `"512 MB"`, `"1.5 MB"`, `"96 KB"`.
///
/// Rounded to the unit that leaves a number a person can hold, because the
/// sentence is read on a status line and the exact byte is not the point.
pub(super) fn shown_bytes(bytes: usize) -> String {
    const KB: usize = 1 << 10;
    const MB: usize = 1 << 20;
    const GB: usize = 1 << 30;
    match bytes {
        b if b >= 10 * GB => format!("{} GB", b / GB),
        b if b >= GB => format!("{:.1} GB", b as f64 / GB as f64),
        b if b >= 10 * MB => format!("{} MB", b / MB),
        b if b >= MB => format!("{:.1} MB", b as f64 / MB as f64),
        b if b >= KB => format!("{} KB", b / KB),
        b => format!("{b} bytes"),
    }
}

impl From<Cancelled> for EvaluateError {
    fn from(_: Cancelled) -> Self {
        EvaluateError::Cancelled
    }
}

impl From<LocalizeError> for EvaluateError {
    fn from(e: LocalizeError) -> Self {
        match e {
            LocalizeError::OutOfMemory { bytes } => EvaluateError::OutOfMemory { bytes },
            LocalizeError::Cancelled => EvaluateError::Cancelled,
        }
    }
}

/// What one evaluation read.
///
/// The counts are of every observation of the track, whatever its verdict: an
/// evaluation reads the `out` ones too, so a refusal is shown beside the number
/// it would have been judged on.
#[derive(Debug, Clone, PartialEq)]
pub struct EvaluateReport {
    /// The stage it ran at.
    pub stage: StageKind,
    /// How many observations came back with a score.
    pub measured: usize,
    /// How many came back with an [`Unmeasured`] reason instead.
    pub unmeasured: usize,
    /// At the cluster stage, the observation the template was cut around. The
    /// kernel picks it -- the largest-scale usable member -- so this is where
    /// it landed, and it is `None` when no member could anchor one.
    pub reference: Option<usize>,
    /// At the track stage, where the track's point stands: what the reading was
    /// made against, rather than anything this step computed. A unit bearing
    /// direction when [`Self::at_infinity`], and a world point otherwise.
    pub position: Option<Point3<f64>>,
    /// At the track stage, whether the track's patch is a bearing (`w == 0`).
    ///
    /// Beside the position because three numbers are not self-describing: the
    /// same triple is a place or a direction depending on this, and printing a
    /// bearing as a position reads as a point one unit from the world origin.
    pub at_infinity: bool,
    /// At the track stage, the condition number of the triangulation the track
    /// carries.
    pub condition_number: Option<f64>,
    /// How many unpinned observations the repaint after the reading turned
    /// `in`. Zero when the track the evaluation read carried a
    /// [`RepaintMark`], because that evaluation does not repaint.
    pub turned_in: usize,
    /// How many it turned `out`.
    pub turned_out: usize,
}

impl std::fmt::Display for EvaluateReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let total = self.measured + self.unmeasured;
        match self.stage {
            StageKind::Cluster => {
                write!(f, "measured {} of {total} observations", self.measured)?;
                match self.reference {
                    Some(reference) => write!(f, " against observation {reference}"),
                    None => write!(f, "; no observation could anchor a template"),
                }
            }
            StageKind::Track => {
                write!(f, "measured {} of {total} observations", self.measured)?;
                match (self.position, self.at_infinity) {
                    (Some(p), false) => write!(f, " at ({:.4}, {:.4}, {:.4})", p.x, p.y, p.z),
                    (Some(p), true) => {
                        write!(f, " along ({:.4}, {:.4}, {:.4}) at infinity", p.x, p.y, p.z)
                    }
                    (None, _) => write!(f, "; the track stands nowhere"),
                }
            }
        }?;
        if self.turned_in + self.turned_out > 0 {
            write!(
                f,
                "; the bars turned {} in and {} out",
                self.turned_in, self.turned_out
            )?;
        }
        Ok(())
    }
}

/// Fill the measurement slots of every observation of `track`, whatever its
/// verdict, at the stage it is in, then let the bars decide the unpinned
/// verdicts from those readings, and change nothing else about it.
///
/// `images` is one [`ProjectedImage`] per image of `edited`, indexed by image
/// index -- the decoded pixels the kernels need, which a reconstruction value
/// does not carry, exactly as every other photometric step of the bench takes
/// them.
///
/// **At the cluster stage** every observation's seed is a member of an
/// in-memory `.matches` cluster, and
/// [`refine_cluster_patches`](crate::patch::cluster_refine::refine_cluster_patches)
/// is run over it: the kernel picks the reference (its largest-scale usable
/// member), cuts the template there, and warps every other seed onto it. What
/// lands in each observation's slot is the refined position and shape, the
/// achieved ZNCC, the drift from the seed, the observation's own tile
/// self-similarity and the kernel's `member_status`, which is that stage's own
/// account of a member it could not fit. The template's half-width is the
/// cluster's own
/// [`ClusterPayload::radius`](super::track::ClusterPayload::radius) rather than
/// the one in `options`, so the square the round registers is the square the
/// seeds were written against. No pose is read.
///
/// **At the track stage** the track's patch is read in every view, at the pixel
/// that view's observation already sits at: one round of the localizer scores
/// each against the leave-one-out consensus of the round's others and finds the
/// correlation peak within the track's
/// [`max_shift_px`](super::track::Thresholds::max_shift_px) of it. What lands in
/// each slot is that ZNCC, how far the peak sits from the observation's own
/// keypoint, how far that keypoint sits from the point's projection, the
/// reprojection error, the ray angle and the tile self-similarity -- and, for an
/// observation no round could read, the [`Unmeasured`] reason instead of a
/// score. **The keypoint itself is not written**, nor the position, the frame or
/// the bitmap: what a reading gives back is the same track with its own account
/// of itself.
///
/// **Then the bars decide every unpinned observation**, in both directions:
/// the readings are painted onto the unpinned rows exactly as
/// [`apply_thresholds`] paints them, best score
/// first and one `in` per image, so an unpinned verdict is always what the bars
/// say about the latest reading. An added observation, which joined `out` with
/// nothing measured, is taken in by the evaluation that first measures it when
/// it clears every bar; a row a step moved past a bar is turned out, and one
/// moved back is turned in. A pinned verdict is never moved.
///
/// **An evaluation that only follows a repaint does not repaint again.** A
/// track-stage reading is scored against the rows that are `in`, so when the
/// repaint changes the `in` set the readings it returns were taken under the
/// old set, and the track returned carries a [`RepaintMark`] saying so. An
/// evaluation of a track that still carries it ([`EditableTrack::repainted`])
/// reads the rows under the new set and leaves every verdict where it is. That
/// bounds the flipping a repaint could otherwise start, a row turned in
/// dropping another below a bar and that one's turn moving the first back,
/// to one extra reading. [`EvaluateReport::turned_in`] and
/// [`EvaluateReport::turned_out`] say what the repaint moved.
///
/// `progress` is where the call names its phases, the names the batch kernels
/// carry: `refine` and `self-similarity` at the cluster stage, `localize` and
/// `self-similarity` at the track stage. Pass `&Progress::none()` to report
/// nothing.
///
/// # Example
///
/// ```no_run
/// # use sfmtool_core::bench::{evaluate, EvaluateOptions};
/// # use sfmtool_core::progress::Progress;
/// # use sfmtool_core::EditedReconstruction;
/// # fn run(
/// #     track: &sfmtool_core::bench::EditableTrack,
/// #     edited: &EditedReconstruction,
/// #     images: &[sfmtool_core::patch::normal_refine::ProjectedImage<'_>],
/// # ) -> Result<(), Box<dyn std::error::Error>> {
/// let (read, report) = evaluate(
///     track,
///     edited,
///     images,
///     &EvaluateOptions::default(),
///     &Progress::none(),
/// )?;
/// println!("{report}");           // "measured 12 of 12 observations at (…)"
/// # let _ = read;
/// # Ok(())
/// # }
/// ```
pub fn evaluate(
    track: &EditableTrack,
    edited: &EditedReconstruction,
    images: &[ProjectedImage<'_>],
    options: &EvaluateOptions,
    progress: &Progress<'_>,
) -> Result<(EditableTrack, EvaluateReport), EvaluateError> {
    check_views(edited, images)?;
    evaluate_preconditions(track)?;
    let (read, mut report) = match &track.stage {
        Stage::Cluster(payload) => evaluate_cluster(track, payload, images, options, progress)?,
        Stage::Track(payload) => {
            let options = options.at_resolution(options.patch_resolution(&edited.base));
            evaluate_track(track, images, payload, &options, progress)?
        }
    };
    if track.repainted() {
        return Ok((read, report));
    }
    let (mut painted, repaint) = apply_thresholds(&read);
    report.turned_in = repaint.turned_in;
    report.turned_out = repaint.turned_out;
    if repaint.changed {
        painted.repaint = RepaintMark::of(&painted);
    }
    Ok((painted, report))
}

/// Whether `track` can be read at the stage it stands in, judged on the track
/// alone.
///
/// The half of [`evaluate`]'s validation that reads no photograph: whether the
/// track stage has a patch to read against. A caller that runs the evaluation
/// somewhere expensive -- on a worker, after decoding a dozen images -- asks
/// this first and refuses in front of the decode, which is a refusal the person
/// who asked for it sees immediately rather than a task that fails a second
/// later.
///
/// There is no minimum `in` count here, because a reading has no minimum: a
/// track with one sighting is read as one sighting with nothing to correlate
/// against, and that is the answer rather than a refusal. A **fit** does have
/// one, and states it in [`fit_preconditions`](super::fit::fit_preconditions).
///
/// [`evaluate`] calls this before anything else it does with the track, so the
/// two cannot come to disagree about what is refused; what is left to
/// [`evaluate`] is everything that needs the views.
///
/// The cluster stage has no such condition: it registers each seed against a
/// template cut from the photographs, so what it can do is a question about the
/// pixels and not about the track.
///
/// # Example
///
/// ```no_run
/// # use sfmtool_core::bench::{evaluate_preconditions, EditableTrack};
/// # fn run(track: &EditableTrack) -> Result<(), Box<dyn std::error::Error>> {
/// evaluate_preconditions(track)?;   // refuse here, before a photograph is read
/// # Ok(())
/// # }
/// ```
pub fn evaluate_preconditions(track: &EditableTrack) -> Result<(), EvaluateError> {
    match &track.stage {
        Stage::Cluster(_) => Ok(()),
        Stage::Track(payload) if payload.placement.is_none() => Err(EvaluateError::NoFrame),
        Stage::Track(_) => Ok(()),
    }
}

/// Every image of `edited` has a decoded view, and there are at least as many
/// views as images: the kernels index `images` by image index.
pub(super) fn check_views(
    edited: &EditedReconstruction,
    images: &[ProjectedImage<'_>],
) -> Result<(), EvaluateError> {
    let expected = edited.image_count();
    if images.len() < expected {
        return Err(EvaluateError::ViewsMissing {
            got: images.len(),
            expected,
        });
    }
    Ok(())
}

/// Every observation, in index order. An `out` observation is read like an
/// `in` one, so the person sees the number the refusal stands beside; only the
/// `in` set decides anything.
pub(super) fn evaluated(track: &EditableTrack) -> Vec<usize> {
    (0..track.observations.len()).collect()
}

/// Every observation names an image the view slice holds.
pub(super) fn check_observation_views(
    track: &EditableTrack,
    images: &[ProjectedImage<'_>],
) -> Result<(), EvaluateError> {
    for observation in &track.observations {
        if observation.image as usize >= images.len() {
            return Err(EvaluateError::NoView {
                image: observation.image,
            });
        }
    }
    Ok(())
}

/// Where an observation currently is, in its image's pixels: the keypoint a
/// track-stage measurement carries, else the cluster stage's refined position
/// or its seed, else nothing.
pub(super) fn seed_of(observation: &Observation) -> Option<[f64; 2]> {
    if let Some(keypoint) = observation.track.as_ref().and_then(|m| m.keypoint) {
        return Some([f64::from(keypoint[0]), f64::from(keypoint[1])]);
    }
    observation.cluster.as_ref().map(|m| m.best_position())
}

// ---- The cluster stage -----------------------------------------------------

/// One seed as the kernel's per-image feature tables hold it: a position and
/// the affine shape at it, in that image's pixels.
type SeedRow = ([f64; 2], [[f64; 2]; 2]);

/// Refine the in-memory cluster every observation's seed makes, and write the
/// kernel's answer into their cluster slots.
///
/// Shared by [`evaluate`] and by [`fit`](super::fit::fit): at the cluster stage
/// the refinement **is** both. There is no geometry to fit -- a cluster is a
/// set of patches registering onto one template, with no point behind them --
/// so the one kernel that reads them writes what it read into the measurement
/// slots and nothing else moves.
pub(super) fn evaluate_cluster(
    track: &EditableTrack,
    payload: &ClusterPayload,
    images: &[ProjectedImage<'_>],
    options: &EvaluateOptions,
    progress: &Progress<'_>,
) -> Result<(EditableTrack, EvaluateReport), EvaluateError> {
    // The cluster's own radius, not the options': the seeds were written
    // against it, so it is what says which square is being registered.
    let params = ClusterRefineParams {
        radius: payload.radius,
        ..options.cluster.clone()
    };
    let mut next = track.clone();
    let mut members: Vec<usize> = Vec::new();
    // The per-image feature tables the kernel reads its seeds out of. The bench
    // holds no `.sift`, so the cluster's own seeds are the detections as far as
    // the kernel is concerned, and a member's "feature index" is its place in
    // its image's list.
    let mut per_image: Vec<Vec<SeedRow>> = vec![Vec::new(); images.len()];
    let mut member_images: Vec<u32> = Vec::new();
    let mut member_features: Vec<u32> = Vec::new();
    for &i in &evaluated(track) {
        let observation = &track.observations[i];
        let image = observation.image as usize;
        if image >= images.len() {
            return Err(EvaluateError::NoView {
                image: observation.image,
            });
        }
        let Some(measurement) = &observation.cluster else {
            // Nothing said where this observation sits in its image, so there
            // is no seed to warp a template onto.
            continue;
        };
        member_images.push(observation.image);
        member_features.push(per_image[image].len() as u32);
        per_image[image].push((measurement.seed_position, measurement.seed_shape));
        members.push(i);
    }

    let positions: Vec<Array2<f32>> = per_image
        .iter()
        .map(|rows| Array2::from_shape_fn((rows.len(), 2), |(k, c)| rows[k].0[c] as f32))
        .collect();
    let shapes: Vec<Array3<f32>> = per_image
        .iter()
        .map(|rows| Array3::from_shape_fn((rows.len(), 2, 2), |(k, r, c)| rows[k].1[r][c] as f32))
        .collect();
    let features: Vec<FeatureGeometry<'_>> = positions
        .iter()
        .zip(&shapes)
        .map(|(p, s)| FeatureGeometry {
            positions_xy: p.view(),
            affine_shapes: s.view(),
        })
        .collect();
    let pyramids: Vec<&ImageU8Pyramid> = images.iter().map(|v| v.pyramid).collect();

    let result = {
        let mut phase = progress.phase("refine");
        let result = refine_cluster_patches_borrowed(
            &pyramids,
            &features,
            &[0, members.len() as u32],
            &member_images,
            &member_features,
            &params,
            None,
        );
        progress_note!(phase, "{} observations", members.len());
        result
    };

    let mut measured = 0;
    for (k, &i) in members.iter().enumerate() {
        let status = result.member_status[k];
        let fitted = matches!(
            status,
            MemberStatus::Reference
                | MemberStatus::Kept
                | MemberStatus::RejectedLowZncc
                | MemberStatus::RejectedShift
        );
        let measurement = next.observations[i]
            .cluster
            .as_mut()
            .expect("every member carries the seed it was built from");
        measurement.position = fitted.then(|| {
            [
                result.member_positions[[k, 0]],
                result.member_positions[[k, 1]],
            ]
        });
        measurement.shape = fitted.then(|| {
            [
                [
                    result.member_affine_shapes[[k, 0, 0]],
                    result.member_affine_shapes[[k, 0, 1]],
                ],
                [
                    result.member_affine_shapes[[k, 1, 0]],
                    result.member_affine_shapes[[k, 1, 1]],
                ],
            ]
        });
        measurement.zncc = finite(f64::from(result.member_zncc[k]));
        measurement.zncc_middle = finite(f64::from(result.member_zncc_middle[k]));
        measurement.zncc_grid = measurement
            .zncc
            .map(|_| result.member_zncc_grid[k].map(|row| row.map(f64::from)));
        // The drift in grid px, the unit of the bar: the refined position's
        // offset from the seed in the seed's keypoint frame, scaled to the
        // template grid.
        measurement.shift_px = measurement
            .position
            .and_then(|position| {
                grid_drift(
                    measurement.seed_position,
                    measurement.seed_shape,
                    position,
                    &params,
                )
            })
            .and_then(finite);
        measurement.status = Some(status);
        if fitted {
            measured += 1;
        }
    }

    // The tile self-similarity, at each member's **seed** geometry, where the
    // kernel cuts the tile it registers.
    {
        let mut phase = progress.phase("self-similarity");
        for &i in &members {
            let observation = &track.observations[i];
            let measurement = observation
                .cluster
                .as_ref()
                .expect("every member carries the seed it was built from");
            let next_measurement = next.observations[i]
                .cluster
                .as_mut()
                .expect("the same member");
            let similarity = tile_self_similarity(
                images[observation.image as usize].pyramid,
                measurement.seed_position,
                measurement.seed_shape,
                &params,
            );
            next_measurement.zncc_self_similarity_radius = similarity.radius;
            next_measurement.zncc_self_similarity_radius_middle = similarity.middle;
            next_measurement.zncc_self_similarity_radius_grid = similarity.grid;
            next_measurement.zncc_self_similarity_ellipse = similarity.ellipse;
            next_measurement.zncc_self_similarity_ellipse_middle = similarity.ellipse_middle;
            next_measurement.zncc_self_similarity_ellipse_grid = similarity.ellipse_grid;
            next_measurement.zncc_self_similarity_surface = similarity.surface;
            next_measurement.zncc_self_similarity_tolerance = similarity.tolerance;
        }
        progress_note!(phase, "{} observations", members.len());
    }

    // Where the kernel cut the template, and the tile it cut.
    let reference = (result.reference_members[0] != REFERENCE_UNREFINABLE)
        .then(|| members[result.reference_members[0] as usize]);
    let template = reference.and_then(|i| {
        let observation = &track.observations[i];
        let measurement = observation.cluster.as_ref()?;
        let grid = sample_member_grid(
            images[observation.image as usize].pyramid,
            measurement.seed_position,
            measurement.seed_shape,
            &params,
        )?;
        let resolution = params.resolution.max(2) as usize;
        let channels = grid.len() / (resolution * resolution);
        Some(ClusterTemplate {
            samples: Array3::from_shape_vec((resolution, resolution, channels), grid)
                .expect("the sampler fills every grid cell of every channel"),
        })
    });
    next.stage = Stage::Cluster(ClusterPayload {
        reference: reference.unwrap_or(payload.reference),
        radius: payload.radius,
        template,
    });

    Ok((
        next,
        EvaluateReport {
            stage: StageKind::Cluster,
            measured,
            unmeasured: evaluated(track).len() - measured,
            reference,
            position: None,
            at_infinity: false,
            condition_number: None,
            turned_in: 0,
            turned_out: 0,
        },
    ))
}

/// One tile's ZNCC self-similarity readings: the radius over the whole core,
/// its middle and each cell of the grid's split, in grid px, their ellipses,
/// and the whole core's ZNCC surface.
#[derive(Default)]
struct TileSelfSimilarity {
    radius: Option<f64>,
    middle: Option<f64>,
    grid: Option<[[f64; 3]; 3]>,
    /// The whole core's ellipse, in grid px, image px and along the patch's
    /// axes, as far as each can be computed.
    ellipse: Option<SelfSimilarityEllipseUnits>,
    /// The same for the middle square.
    ellipse_middle: Option<SelfSimilarityEllipseUnits>,
    /// Each cell's ellipse, in grid px.
    ellipse_grid: Option<[[SelfSimilarityEllipse; 3]; 3]>,
    surface: Option<Vec<f64>>,
    tolerance: Option<f64>,
}

/// What a tile's grid is, for converting its self-similarity readings out of
/// grid px: the image px per grid px at the tile's centre, and the placement
/// the `R×R` tile was rendered through, where there is one.
struct TileGeometry<'a> {
    jacobian: Option<[[f64; 2]; 2]>,
    placement: Option<&'a OrientedPatch>,
}

/// Read an interleaved `R × R × C` tile's self-similarity the overlap way,
/// with the default parameters, and measure the whole tile's and the middle's
/// ellipses through `geometry`.
///
/// The tile is read as it is, every sample as data, with no pixels from
/// outside it: at each shift only the samples both windows hold are
/// correlated, as the culls read a stored bitmap.
fn score_self_similarity(
    samples: &[f32],
    channels: usize,
    resolution: usize,
    geometry: &TileGeometry<'_>,
) -> TileSelfSimilarity {
    if channels == 0 || resolution < 3 || samples.len() != resolution * resolution * channels {
        return TileSelfSimilarity::default();
    }
    let (planes, colour) =
        PatchTile::planes_from_interleaved(samples, resolution, resolution, channels);
    let tile = PatchTile {
        values: &planes,
        channels: colour,
        width: resolution,
        height: resolution,
    };
    let parts = zncc_self_similarity_parts(&tile, None, &SelfSimilarityParams::default());
    let units = |reading| {
        SelfSimilarityEllipseUnits::read(reading, geometry.jacobian, geometry.placement, resolution)
    };
    TileSelfSimilarity {
        ellipse: units(&parts.whole),
        ellipse_middle: units(&parts.middle),
        radius: Some(parts.whole.radius),
        middle: Some(parts.middle.radius),
        grid: Some(
            parts
                .grid
                .each_ref()
                .map(|row| row.each_ref().map(|cell| cell.radius)),
        ),
        ellipse_grid: Some(
            parts
                .grid
                .each_ref()
                .map(|row| row.each_ref().map(|cell| cell.ellipse)),
        ),
        tolerance: finite(parts.whole.tolerance),
        surface: Some(parts.whole.surface),
    }
}

/// One observation's own tile self-similarity, at its seed geometry: the
/// member's `R×R` grid ([`sample_member_grid`]), which cluster refinement's
/// member gate reads, so the whole radius here is the number that gate
/// judges. All `None` when the geometry is degenerate or the tile leaves the
/// pyramid.
fn tile_self_similarity(
    pyramid: &ImageU8Pyramid,
    position: [f64; 2],
    shape: [[f64; 2]; 2],
    params: &ClusterRefineParams,
) -> TileSelfSimilarity {
    let Some(tile) = sample_member_grid(pyramid, position, shape, params) else {
        return TileSelfSimilarity::default();
    };
    let resolution = params.resolution.max(2) as usize;
    let channels = tile.len() / (resolution * resolution);
    // The grid is the seed shape's affine map: one grid px is
    // `2 · radius / R` keypoint-frame units, which the shape carries to image
    // px, columns along the shape's first column.
    let step = 2.0 * params.radius / resolution as f64;
    let geometry = TileGeometry {
        jacobian: Some(shape.map(|row| row.map(|v| v * step))),
        placement: None,
    };
    score_self_similarity(&tile, channels, resolution, &geometry)
}

// ---- The track stage -------------------------------------------------------

/// The rounds one pass over a track's observations takes, and the observations
/// that are in none of them.
///
/// **One round holds one observation per image**, because the localizer
/// registers a *point's* sighting in a view and two sightings in one view are
/// two hypotheses about that view. So the first round is the `in` observations
/// plus every other observation in an image none of them holds, which is the
/// shape the localizer is meant to run over; an
/// observation in an image that is already spoken for gets a round of its own,
/// against the `in` observations minus the one whose image it wants, which is
/// the same leave-one-out question asked about the other hypothesis.
pub(super) struct Rounds {
    /// The first round, `in` observations first.
    pub first: Vec<usize>,
    /// The observations that each get a round of their own.
    pub contested: Vec<usize>,
    /// The observations in no round at all, each with the reason, which is a
    /// property of the observation and the geometry rather than of any
    /// correlation.
    pub excluded: Vec<(usize, Unmeasured)>,
}

impl Rounds {
    /// The round one contested observation is read in: the first round's `in`
    /// observations minus the one whose image it wants, plus itself.
    pub(super) fn contested_round(&self, track: &EditableTrack, i: usize) -> Vec<usize> {
        let image = track.observations[i].image;
        let mut round: Vec<usize> = self
            .first
            .iter()
            .copied()
            .filter(|&j| {
                track.observations[j].verdict == Verdict::In && track.observations[j].image != image
            })
            .collect();
        round.push(i);
        round
    }
}

/// Plan the rounds of one pass over `track` against `frame`, and name what each
/// excluded observation was excluded for.
///
/// The five exclusions are the localizer's own up-front refusals, decided here
/// so that the row carries the reason rather than simply going missing from the
/// kernel's answer: no seed at all, a seed off the sensor, a point that does not
/// project into the view, a ray that grazes the patch plane, and a seed further
/// from the projection than `max_seed_offset_px` patch-grid px.
///
/// **The offset bound is a memory bound.** A round's search window is widened to
/// reach its furthest seed and each view's tile is `resolution + 4 · window` on
/// a side, so one far-out observation sizes every tile of its round and the cost
/// is that offset squared per view. Deciding it here means the tile is never
/// asked for: the row is left out of the round carrying
/// [`Unmeasured::SeedTooFar`], and the observations that can be read are read.
pub(super) fn plan_rounds(
    track: &EditableTrack,
    images: &[ProjectedImage<'_>],
    frame: &OrientedPatch,
    localize: &KeypointLocalizeParams,
    max_seed_offset_px: f64,
) -> Rounds {
    let min_grazing_cos = localize.min_grazing_cos;
    let normal = frame.normal();
    let mut excluded: Vec<(usize, Unmeasured)> = Vec::new();
    let readable = |i: usize| -> Result<(), Unmeasured> {
        let observation = &track.observations[i];
        let view = &images[observation.image as usize];
        let Some(seed) = seed_of(observation) else {
            return Err(Unmeasured::NoSeed);
        };
        if !(seed[0] >= 0.0
            && seed[1] >= 0.0
            && seed[0] < f64::from(view.camera.width)
            && seed[1] < f64::from(view.camera.height))
        {
            return Err(Unmeasured::OffSensor);
        }
        // The viewing direction is camera-to-point for a finite point, and the
        // direction itself for one at infinity, exactly as the localizer reads
        // it.
        let d = if frame.w == 0.0 {
            frame.center.coords
        } else {
            frame.center - view.cam_from_world.inverse_translation_origin()
        };
        let norm = d.norm();
        let cosine = if norm > 1e-12 {
            (d.dot(&normal) / norm).abs()
        } else {
            0.0
        };
        if norm <= 1e-12 || cosine < min_grazing_cos {
            return Err(Unmeasured::Grazing { cosine });
        }
        if project_unclipped(view, &frame.center, frame.w).is_none() {
            return Err(Unmeasured::NoProjection);
        }
        // How far the window would have to be widened to reach this seed, and
        // whether that is a widening worth making. A seed the kernel cannot
        // measure an offset for at all -- the same refusals the seeding itself
        // makes -- is left to the round, which seeds it at the projection.
        if let Some(off) = keypoint_grid_offset(frame, view, seed, localize) {
            let offset_px = off[0].hypot(off[1]);
            if offset_px.is_finite() && offset_px > max_seed_offset_px {
                return Err(Unmeasured::SeedTooFar {
                    offset_px,
                    bound_px: max_seed_offset_px,
                });
            }
        }
        Ok(())
    };

    let mut first: Vec<usize> = Vec::new();
    let mut taken: Vec<u32> = Vec::new();
    for &i in &track.in_observations() {
        match readable(i) {
            Ok(()) => {
                taken.push(track.observations[i].image);
                first.push(i);
            }
            Err(why) => excluded.push((i, why)),
        }
    }
    let mut contested: Vec<usize> = Vec::new();
    for i in evaluated(track) {
        if track.observations[i].verdict == Verdict::In {
            continue;
        }
        if let Err(why) = readable(i) {
            excluded.push((i, why));
            continue;
        }
        let image = track.observations[i].image;
        if taken.contains(&image) {
            contested.push(i);
        } else {
            taken.push(image);
            first.push(i);
        }
    }
    Rounds {
        first,
        contested,
        excluded,
    }
}

/// What one round's localizer said about one observation, read at the pixel the
/// observation already sits at.
struct Reading {
    /// The leave-one-out ZNCC at the correlation peak.
    zncc: f64,
    /// The same reading over the middle of the tile.
    zncc_middle: f64,
    /// The same reading over each cell of the ZNCC grid.
    zncc_grid: [[f64; 3]; 3],
    /// How far that peak sits from the observation's own keypoint, in
    /// source-image px.
    seed_shift_px: f64,
}

/// Read every observation of `track` against the patch it carries, and write
/// what each one says about itself into its track-stage slot.
fn evaluate_track(
    track: &EditableTrack,
    images: &[ProjectedImage<'_>],
    payload: &TrackPayload,
    options: &EvaluateOptions,
    progress: &Progress<'_>,
) -> Result<(EditableTrack, EvaluateReport), EvaluateError> {
    let frame = payload.placement.clone().ok_or(EvaluateError::NoFrame)?;
    check_observation_views(track, images)?;

    let plan = plan_rounds(
        track,
        images,
        &frame,
        &options.localize,
        options.max_seed_offset_px,
    );
    let mut readings: HashMap<usize, Reading> = HashMap::new();
    let mut reasons: HashMap<usize, Unmeasured> = plan.excluded.iter().copied().collect();
    {
        let mut phase = progress.phase("localize");
        let mut rounds = 0;
        read_round(
            track,
            images,
            &frame,
            &plan.first,
            options,
            progress,
            &mut readings,
            &mut reasons,
        )?;
        rounds += 1;
        for &i in &plan.contested {
            progress.check_cancel()?;
            let round = plan.contested_round(track, i);
            read_round(
                track,
                images,
                &frame,
                &round,
                options,
                progress,
                &mut readings,
                &mut reasons,
            )?;
            rounds += 1;
        }
        progress_note!(phase, "{rounds} rounds, {} observations", readings.len());
    }

    // Per observation: what the correlation said, and what the geometry says
    // about where it sits. The second half is computed for every row that has a
    // pixel at all, read or not -- an observation the localizer could not score
    // still has a distance from the projection, and that distance is often the
    // thing that explains the row.
    let resolution = options.localize.resolution.max(2) as usize;
    let mut next = track.clone();
    let (mut measured, mut unmeasured) = (0, 0);
    // The `in` observations' tiles, kept for the reference view's readings,
    // which correlate them in pairs.
    let mut kept_tiles: Vec<(usize, ViewTile)> = Vec::new();
    {
        let mut phase = progress.phase("self-similarity");
        for i in evaluated(track) {
            let observation = &track.observations[i];
            let view = &images[observation.image as usize];
            let mut measurement = observation.track.clone().unwrap_or_default();
            let reading = readings.get(&i);
            measurement.zncc = reading.and_then(|r| finite(r.zncc));
            measurement.zncc_middle = measurement
                .zncc
                .and(reading.and_then(|r| finite(r.zncc_middle)));
            measurement.zncc_grid = measurement.zncc.and(reading.map(|r| r.zncc_grid));
            measurement.seed_shift_px = reading.and_then(|r| finite(r.seed_shift_px));
            measurement.reason = match measurement.zncc {
                Some(_) => None,
                None => Some(reasons.get(&i).copied().unwrap_or(Unmeasured::Unscorable)),
            };
            measurement.projection_offset_px = None;
            measurement.reprojection_error = None;
            measurement.ray_angle_deg = None;
            measurement.zncc_self_similarity_radius = None;
            measurement.zncc_self_similarity_radius_middle = None;
            measurement.zncc_self_similarity_radius_grid = None;
            measurement.zncc_self_similarity_ellipse = None;
            measurement.zncc_self_similarity_ellipse_middle = None;
            measurement.zncc_self_similarity_ellipse_grid = None;
            measurement.zncc_self_similarity_surface = None;
            measurement.zncc_self_similarity_tolerance = None;
            measurement.viewing_angle_deg = None;
            measurement.tilt_direction_deg = None;
            measurement.coverage = None;
            measurement.clipped_share = None;
            measurement.pair_zncc = None;
            measurement.pair_zncc_grid = None;
            measurement.cell_deficit = None;
            measurement.reference_view = None;
            if let Some(pixel) = seed_of(observation) {
                // The offset is measured from the **patch's** projection,
                // because that is the anchor the localizer renders its tile
                // about and the point the seed offset above was clipped
                // against; the reprojection error below is against the
                // triangulated position, which is the residual a commit
                // stores. For a fitted track the two are one point.
                if let Some((u, v)) = project_unclipped(view, &frame.center, frame.w) {
                    measurement.projection_offset_px = finite((u - pixel[0]).hypot(v - pixel[1]));
                }
                if let Some(position) = payload.position {
                    let (error, angle) = observation_metrics(view, &position, frame.w, pixel);
                    measurement.reprojection_error = finite(error);
                    measurement.ray_angle_deg = finite(angle);
                }
                let tile = render_view_tile(
                    &frame,
                    view,
                    Some(pixel),
                    resolution,
                    options.localize.sampler,
                    &phase,
                );
                let similarity = view_tile_self_similarity(&tile);
                measurement.zncc_self_similarity_radius = similarity.radius;
                measurement.zncc_self_similarity_radius_middle = similarity.middle;
                measurement.zncc_self_similarity_radius_grid = similarity.grid;
                measurement.zncc_self_similarity_ellipse = similarity.ellipse;
                measurement.zncc_self_similarity_ellipse_middle = similarity.ellipse_middle;
                measurement.zncc_self_similarity_ellipse_grid = similarity.ellipse_grid;
                measurement.zncc_self_similarity_surface = similarity.surface;
                measurement.zncc_self_similarity_tolerance = similarity.tolerance;
                measurement.viewing_angle_deg = tile.viewing_angle.map(|a| a.angle_deg);
                measurement.tilt_direction_deg =
                    tile.viewing_angle.and_then(|a| a.tilt_direction_deg);
                measurement.coverage = Some(tile.coverage);
                measurement.clipped_share = tile.clipped_share;
                if observation.verdict == Verdict::In {
                    kept_tiles.push((i, tile));
                }
            }
            if measurement.zncc.is_some() {
                measured += 1;
            } else {
                unmeasured += 1;
            }
            next.observations[i].track = Some(measurement);
        }
        progress_note!(phase, "{} observations", track.observations.len());
    }
    progress.check_cancel()?;
    read_reference_view(&mut next, images, &frame, &kept_tiles, options, progress);

    Ok((
        next,
        EvaluateReport {
            stage: StageKind::Track,
            measured,
            unmeasured,
            reference: None,
            position: payload.position,
            // The payload's flag, not the frame's `w`: the two agree wherever a
            // frame exists, and the flag is the one a frameless track also
            // carries.
            at_infinity: payload.at_infinity,
            condition_number: payload.condition_number,
            turned_in: 0,
            turned_out: 0,
        },
    ))
}

/// Score one round's observations against each other, where they sit.
///
/// One round of the localizer: the cores are read at the observations' own
/// keypoints, each is scored against the leave-one-out consensus of the others,
/// and the peak of its shift search is where the correlation says it would
/// rather be. Nothing is written back to the track here -- the peak is reported
/// as a distance, not taken.
#[allow(clippy::too_many_arguments)]
fn read_round(
    track: &EditableTrack,
    images: &[ProjectedImage<'_>],
    frame: &OrientedPatch,
    round: &[usize],
    options: &EvaluateOptions,
    progress: &Progress<'_>,
    readings: &mut HashMap<usize, Reading>,
    reasons: &mut HashMap<usize, Unmeasured>,
) -> Result<(), EvaluateError> {
    if round.len() < 2 {
        for &i in round {
            reasons.entry(i).or_insert(Unmeasured::NoConsensus);
        }
        return Ok(());
    }
    let view_set: Vec<u32> = round.iter().map(|&i| track.observations[i].image).collect();
    let seeds: Vec<Option<[f64; 2]>> = round
        .iter()
        .map(|&i| seed_of(&track.observations[i]))
        .collect();
    let of_image: HashMap<u32, usize> = view_set
        .iter()
        .copied()
        .zip(round.iter().copied())
        .collect();
    let at_image: HashMap<u32, usize> = view_set.iter().copied().zip(0..round.len()).collect();

    let params = KeypointLocalizeParams {
        search: search_radius(track, images, frame, round, options),
        ..options.localize.clone()
    };
    // What this round would cost, before a byte of it is asked for: the tile
    // each view renders at the widened window, summed. A round past the budget
    // is refused rather than attempted, because the allocation that would fail
    // is the one that aborts the process.
    let bytes = round_cache_bytes(track, images, round, &params);
    if bytes > options.max_cache_bytes {
        return Err(EvaluateError::TooLarge {
            bytes,
            budget: options.max_cache_bytes,
        });
    }
    let localized =
        try_localize_patch_keypoints(frame, images, &view_set, Some(&seeds), &params, progress)?;

    for (slot, &image) in localized.views.iter().enumerate() {
        let (Some(&i), Some(&at)) = (of_image.get(&image), at_image.get(&image)) else {
            continue;
        };
        // The first round's numbers are the ones a row keeps: a contested round
        // re-reads the `in` observations against a consensus one of them was
        // left out of, which answers a question about the *other* hypothesis.
        if readings.contains_key(&i) {
            continue;
        }
        let peak = localized.keypoints[slot];
        // The shift is read on the patch's plane, in grid px, the unit of the
        // bar and of the self-similarity radius: both ends through the same
        // unprojection the localizer seeds from.
        let view = &images[image as usize];
        let shift = match seeds[at] {
            Some(seed) => grid_distance(frame, view, seed, peak, &params),
            None => f64::NAN,
        };
        let zncc = localized.loo_zncc[slot];
        if !zncc.is_finite() {
            // The kernel gave the view back without ever scoring it, which is
            // what it does when too few of the round's views could be read
            // together for a consensus to exist. The row keeps its distances
            // and says that, rather than reading as an unexplained blank.
            reasons.entry(i).or_insert(Unmeasured::NoConsensus);
        }
        readings.insert(
            i,
            Reading {
                zncc,
                zncc_middle: localized.loo_zncc_middle[slot],
                zncc_grid: localized.loo_zncc_grid[slot],
                seed_shift_px: shift,
            },
        );
    }
    for &i in round {
        if !readings.contains_key(&i) {
            reasons.entry(i).or_insert(Unmeasured::Unscorable);
        }
    }
    Ok(())
}

/// What one round's per-view tiles cost together, in bytes.
///
/// One [`view_cache_bytes`] per observation of the round, at that round's own
/// widened window and its own photograph's channel count. This is the number
/// [`EvaluateOptions::max_cache_bytes`] is a budget on, and it is computed from
/// the parameters alone -- nothing is rendered to find it out.
pub(super) fn round_cache_bytes(
    track: &EditableTrack,
    images: &[ProjectedImage<'_>],
    round: &[usize],
    params: &KeypointLocalizeParams,
) -> usize {
    round
        .iter()
        .map(|&i| {
            let view = &images[track.observations[i].image as usize];
            let channels = view.pyramid.level(0).channels() as usize;
            view_cache_bytes(params, channels)
        })
        .fold(0usize, |total, bytes| total.saturating_add(bytes))
}

/// The distance from `from` to `to`, two pixels of `view`, on `frame`'s plane
/// in patch-grid px, or `NaN` where either does not reach the plane.
pub(super) fn grid_distance(
    frame: &OrientedPatch,
    view: &ProjectedImage<'_>,
    from: [f64; 2],
    to: [f64; 2],
    params: &KeypointLocalizeParams,
) -> f64 {
    match (
        keypoint_grid_offset(frame, view, from, params),
        keypoint_grid_offset(frame, view, to, params),
    ) {
        (Some(a), Some(b)) => (b[0] - a[0]).hypot(b[1] - a[1]),
        _ => f64::NAN,
    }
}

/// The cluster stage's drift from `seed` to `position`, both in source-image
/// px, in patch-grid px: through the inverse of the seed's shape (keypoint-frame
/// units to pixels), then `resolution` grid px across `2 · radius` units.
/// `None` for a singular shape.
fn grid_drift(
    seed: [f64; 2],
    shape: [[f64; 2]; 2],
    position: [f64; 2],
    params: &ClusterRefineParams,
) -> Option<f64> {
    let det = shape[0][0] * shape[1][1] - shape[0][1] * shape[1][0];
    if det == 0.0 || !det.is_finite() || params.radius <= 0.0 {
        return None;
    }
    let d = [position[0] - seed[0], position[1] - seed[1]];
    let u = (shape[1][1] * d[0] - shape[0][1] * d[1]) / det;
    let v = (-shape[1][0] * d[0] + shape[0][0] * d[1]) / det;
    Some(u.hypot(v) * f64::from(params.resolution) / (2.0 * params.radius))
}

/// The search radius one round runs at, in patch-grid px: the track's
/// [`max_shift_px`](super::track::Thresholds::max_shift_px) plus the furthest
/// seed's own offset from the point's projection.
///
/// The bar is the radius because it is the same question: how far from where a
/// sighting sits the correlation may put it. A peak the bar would refuse is
/// one this window can still find, at its edge or past it.
///
/// The kernel anchors its window at that projection and clips a seed beyond
/// `search` back onto the bound, so a window sized for the search radius alone
/// would start a far-out observation short of where it sits and report the
/// correlation of a place the sighting is not. Widening by the furthest offset
/// is what lets every observation be read where it actually is; in return, an
/// observation in a round that holds a far-out seed may report a peak further
/// than the bar from itself, which is the honest reading of a window that had
/// to be that wide.
///
/// The widening is bounded, and the bound is applied a step earlier: a seed past
/// [`EvaluateOptions::max_seed_offset_px`] is not in the round at all
/// ([`plan_rounds`]), so the widest offset this can find is that bound.
fn search_radius(
    track: &EditableTrack,
    images: &[ProjectedImage<'_>],
    frame: &OrientedPatch,
    round: &[usize],
    options: &EvaluateOptions,
) -> f64 {
    let mut widest: f64 = 0.0;
    for &i in round {
        let observation = &track.observations[i];
        let Some(seed) = seed_of(observation) else {
            continue;
        };
        let view = &images[observation.image as usize];
        if let Some(off) = keypoint_grid_offset(frame, view, seed, &options.localize) {
            let distance = off[0].hypot(off[1]);
            if distance.is_finite() {
                widest = widest.max(distance);
            }
        }
    }
    track.thresholds.max_shift_px.max(0.0) + widest
}

/// One observation's reprojection error in px and the angle its ray makes with
/// the direction to `position`, in degrees -- the two numbers the viewer's
/// Track View tabulates for a committed track, over this track's own position
/// and the pixel this observation sits at rather than the stored ones.
///
/// `w` says which `position` is. At `w == 1` the angle is between the sighting's
/// ray and the camera-to-point direction; at `w == 0` `position` is a world
/// bearing and the angle is between the sighting's ray and that bearing, which
/// is the only thing either of them states.
pub(super) fn observation_metrics(
    view: &ProjectedImage<'_>,
    position: &Point3<f64>,
    w: f64,
    keypoint: [f64; 2],
) -> (f64, f64) {
    let Some((u, v)) = project_unclipped(view, position, w) else {
        return (f64::NAN, f64::NAN);
    };
    let error = (u - keypoint[0]).hypot(v - keypoint[1]);
    let ray = view.camera.pixel_to_ray(keypoint[0], keypoint[1]);
    let ray = Vector3::new(ray[0], ray[1], ray[2]);
    // Homogeneous, because `w` is the whole difference between a place and a
    // direction: the camera-frame vector toward a finite point carries the
    // pose's translation, and the one toward a bearing does not. Applying the
    // translation to a unit direction would measure the angle to a phantom point
    // one unit from the world origin, which is the one thing a bearing is not.
    let towards = view
        .cam_from_world
        .transform_point_homogeneous(position.coords, w);
    let (ray_norm, towards_norm) = (ray.norm(), towards.norm());
    if !(ray_norm > 0.0 && towards_norm > 0.0) {
        return (error, f64::NAN);
    }
    let cos = (ray.dot(&towards) / (ray_norm * towards_norm)).clamp(-1.0, 1.0);
    (error, cos.acos().to_degrees())
}

/// The self-similarity of what one view shows of the patch at its keypoint.
///
/// It reads the view's `R×R` tile ([`render_view_tile`], the grid every
/// stored patch bitmap is rendered on), the overlap way, with no pixels from
/// outside it. The ellipses are measured through the placement the tile was
/// rendered through at resolution `R`: the image px per grid px at its centre
/// and its half-extents.
fn view_tile_self_similarity(tile: &ViewTile) -> TileSelfSimilarity {
    let samples: Vec<f32> = tile.samples.iter().map(|&v| f32::from(v)).collect();
    let geometry = TileGeometry {
        jacobian: tile.jacobian,
        placement: Some(&tile.placement),
    };
    score_self_similarity(&samples, tile.channels(), tile.resolution(), &geometry)
}

/// Read what the reference-view rule needs across the `in` observations whose
/// tiles are in `tiles`, run the rule, and write each one's pair ZNCC, pair ZNCC
/// grid, cell deficit and standing into its track-stage slot.
///
/// The pair ZNCC is the median of the observation's row of member coherence's
/// matrix ([`member_zncc_matrix_reporting`]), rendered over the observations'
/// common support with member coherence's window at the evaluation's
/// resolution and sampler, anchored at each observation's keypoint. The cell
/// readings come from the tiles already rendered ([`cell_agreement`]): member
/// coherence renders only the samples inside its window's disk and common to
/// every member, so its corner cells would hold part of their square, and the
/// cell check was measured on whole cells with each pair's own support.
/// Timed as the `reference view` phase of `progress`.
fn read_reference_view(
    next: &mut EditableTrack,
    images: &[ProjectedImage<'_>],
    frame: &OrientedPatch,
    tiles: &[(usize, ViewTile)],
    options: &EvaluateOptions,
    progress: &Progress<'_>,
) {
    if tiles.is_empty() {
        return;
    }
    let mut phase = progress.phase("reference view");
    let members: Vec<u32> = tiles
        .iter()
        .map(|(i, _)| next.observations[*i].image)
        .collect();
    let keypoints: Vec<Option<[f64; 2]>> = tiles
        .iter()
        .map(|(i, _)| seed_of(&next.observations[*i]))
        .collect();
    let params = MemberCoherenceParams {
        resolution: options.localize.resolution.max(2),
        sampler: options.localize.sampler,
        ..MemberCoherenceParams::default()
    };
    let matrix =
        member_zncc_matrix_reporting(frame, images, &members, Some(&keypoints), &params, progress);
    // One `in` observation per image, so the matrix's members are the tiles'
    // images in the same order; looked up by image all the same.
    let pair_zncc: Vec<Option<f64>> = members
        .iter()
        .map(|image| {
            let row = matrix.members.iter().position(|m| m == image)?;
            let others: Vec<f64> = (0..matrix.len())
                .filter(|&j| j != row)
                .map(|j| matrix.get(row, j))
                .collect();
            finite(finite_middle(&others))
        })
        .collect();
    let refs: Vec<&ViewTile> = tiles.iter().map(|(_, tile)| tile).collect();
    let cells = cell_agreement(&refs);
    let readings: Vec<ReferenceReadings> = tiles
        .iter()
        .enumerate()
        .map(|(k, (i, tile))| {
            let axes = next.observations[*i]
                .track
                .as_ref()
                .and_then(|m| m.zncc_self_similarity_ellipse)
                .map(|e| e.grid_px.axes);
            ReferenceReadings {
                coverage: Some(tile.coverage),
                clipped_share: tile.clipped_share,
                viewing_angle_deg: tile.viewing_angle.map(|a| a.angle_deg),
                cell_deficit: finite(cells.deficit[k]),
                pair_zncc: pair_zncc[k],
                semi_major: axes.map(|a| a[0]),
                semi_minor: axes.map(|a| a[1]),
            }
        })
        .collect();
    let choice = choose_reference_view(&readings);
    for (k, (i, _)) in tiles.iter().enumerate() {
        if let Some(measurement) = next.observations[*i].track.as_mut() {
            measurement.pair_zncc = pair_zncc[k];
            measurement.pair_zncc_grid = Some(cells.pair_zncc_grid[k]);
            measurement.cell_deficit = finite(cells.deficit[k]);
            measurement.reference_view = choice.standing(k);
        }
    }
    match choice.reference {
        Some(k) => progress_note!(
            phase,
            "{} views, observation {} picked",
            tiles.len(),
            tiles[k].0
        ),
        None => progress_note!(phase, "{} views, none picked", tiles.len()),
    }
}

/// `Some(value)` when it is a number, `None` when the kernel reported nothing.
///
/// A `NaN` out of a kernel means "this was not scored", and the slots say that
/// with `None`; a `NaN` left in a slot would read as a measured failure to the
/// painting, which is a different thing entirely.
pub(super) fn finite(value: f64) -> Option<f64> {
    value.is_finite().then_some(value)
}
