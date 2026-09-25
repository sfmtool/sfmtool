// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Adding one image's observations to the tracks it can see.
//!
//! After an image has been given a pose it did not have (by *Resect Image*, or
//! by hand), it still observes only the tracks it observed before. This asks,
//! of every point the image does not observe, whether the image sees the point,
//! where, and whether what it sees there agrees with the point's other
//! observations; and it adds the observations that pass. The existing
//! observations are the reference and do not move, and neither does anything
//! else: no point, no frame, no bitmap, no camera. Nothing is re-triangulated or
//! adjusted.
//!
//! The kernels are the localizer's: the point's existing observations are
//! rendered where their keypoints put them and combined into a robust
//! consensus ([`ReferenceConsensus`]), the image is searched once against it,
//! optionally refined to sub-pixel against the same references
//! ([`refine_view_against_references`]), and scored. See
//! `specs/core/reconstruction/add-image-to-tracks.md` for the design, and
//! `scripts/add_image_to_tracks/README.md` for the evaluation that chose the
//! default rule.

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};

use nalgebra::{Point3, Vector3};
use ndarray::Array2;
use rayon::prelude::*;

use sfmtool_sfmr_format::ContentHash;

use crate::camera::remap::ImageU8Pyramid;
use crate::geometry::RigidTransform;
use crate::numeric::median_in_place;
use crate::patch::cloud::OrientedPatch;
use crate::patch::keypoint_localize::{
    project_unclipped, KeypointLocalizeParams, ReferenceConsensus,
};
use crate::patch::keypoint_subpixel::{refine_view_against_references, KeypointSubpixelParams};
use crate::patch::normal_refine::ProjectedImage;
use crate::progress::{Cancelled, Progress};
use crate::progress_info;
use crate::reconstruction::bundle_adjust::is_posed;
use crate::reconstruction::data::{ObservationSource, SfmrReconstruction, TrackObservation};

/// The scale factor from a median absolute deviation to a normal standard
/// deviation.
const MAD_TO_SIGMA: f64 = 1.4826;

/// Which statistic of a set of reference ZNCCs sets a bar.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BasisStatistic {
    /// The smallest: the new view must agree at least as well as the worst
    /// existing one.
    Min,
    /// The median minus `k` times the scaled median absolute deviation.
    MedianMinusMad {
        /// How many scaled deviations below the median the bar sits.
        k: f64,
    },
    /// A fraction of the median.
    FractionOfMedian {
        /// The fraction.
        fraction: f64,
    },
}

impl BasisStatistic {
    /// The bar this statistic sets over `values` (`NaN` when empty).
    pub fn bar(&self, values: &[f64]) -> f64 {
        let mut v: Vec<f64> = values.iter().copied().filter(|x| x.is_finite()).collect();
        if v.is_empty() {
            return f64::NAN;
        }
        match *self {
            BasisStatistic::Min => v.iter().copied().fold(f64::INFINITY, f64::min),
            BasisStatistic::MedianMinusMad { k } => {
                let med = median_in_place(&mut v);
                let mut dev: Vec<f64> = v.iter().map(|x| (x - med).abs()).collect();
                let mad = median_in_place(&mut dev);
                med - k * MAD_TO_SIGMA * mad
            }
            BasisStatistic::FractionOfMedian { fraction } => fraction * median_in_place(&mut v),
        }
    }
}

/// Which of the new view's two pairwise ZNCCs a two-reference track judges.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PairStatistic {
    /// The smaller: the new view must agree with both references.
    Min,
    /// Their mean.
    Mean,
    /// The larger: the new view must agree with one of them.
    Max,
}

/// The rule for a track with exactly two references: a statistic of the new
/// view's ZNCC against each reference must reach `factor` times the ZNCC
/// between the two references.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PairRule {
    /// Which of the new view's two pairwise ZNCCs is judged.
    pub statistic: PairStatistic,
    /// The multiple of the references' own pairwise ZNCC it must reach.
    pub factor: f64,
}

/// How a candidate's photometric agreement is judged.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AcceptRule {
    /// Accept at or above [`AddImageToTracksOptions::min_zncc`] and nothing
    /// else.
    FixedZncc,
    /// The point's own references set the bar: with three or more, a statistic
    /// of their leave-one-out ZNCCs; with two, the pair rule.
    TrackBasis {
        /// The statistic over three or more references' leave-one-out ZNCCs.
        statistic: BasisStatistic,
        /// The rule for exactly two references.
        pair: PairRule,
    },
    /// One bar for the whole call: the statistic over the leave-one-out ZNCCs
    /// of every reference of every candidate point that reached the verdict.
    PooledBasis {
        /// The statistic over the pooled leave-one-out ZNCCs.
        statistic: BasisStatistic,
    },
    /// Accept a candidate that reaches either bar: the pooled one, or the one
    /// its own track sets. A track whose references agree less well with each
    /// other than the image's tracks do on the whole sets a lower bar of its
    /// own, and a sighting as good as its references is not refused for being
    /// on a harder surface than the rest.
    PooledOrTrack {
        /// The statistic over the pooled leave-one-out ZNCCs.
        pooled: BasisStatistic,
        /// The statistic over three or more references' leave-one-out ZNCCs.
        track: BasisStatistic,
        /// The rule for exactly two references.
        pair: PairRule,
    },
}

/// How far from the point's projection a keypoint may land.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PositionGate {
    /// No positional gate.
    Off,
    /// At most this many source pixels.
    MaxPx(f64),
    /// Derived from the call's own photometrically accepted candidates: at most
    /// their median offset plus `k` scaled median absolute deviations, and
    /// never less than `floor_px`. With fewer than three such candidates no
    /// bound is derived and none is applied.
    ImageMad {
        /// How many scaled deviations above the median the bound sits.
        k: f64,
        /// The smallest bound, in source pixels.
        floor_px: f64,
    },
}

/// What the new view is searched against.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TemplateSource {
    /// The robust consensus of the references, rendered at their keypoints.
    Rendered,
    /// The point's stored patch bitmap, where the value has one for it; the
    /// rendered consensus where it does not.
    StoredBitmap,
}

/// The rules one call judges each candidate by.
#[derive(Debug, Clone)]
pub struct AddImageToTracksOptions {
    /// The photometric rule.
    pub rule: AcceptRule,
    /// A ZNCC floor every rule applies to the new view's ZNCC against the
    /// consensus, and the whole of [`AcceptRule::FixedZncc`]. `0` or below
    /// disables it for the other rules.
    pub min_zncc: f64,
    /// The positional gate.
    pub position_gate: PositionGate,
    /// What the new view is searched against.
    pub template: TemplateSource,
    /// Refuse a finite point whose patch plane has the new camera on the other
    /// side from the majority of the cameras that observe it.
    pub require_facing: bool,
    /// Refine the searched keypoint to sub-pixel against the references.
    pub subpixel: bool,
    /// Where the search window's highest peak is on its edge, take the local
    /// maximum an ascent from the projection reaches instead, when that one is
    /// inside the window. See [`ReferenceConsensus::search`].
    pub ascend_on_edge: bool,
    /// Two keypoints in the new image closer than this, in source pixels, are
    /// one place: of two accepted candidates the one with the lower ZNCC is
    /// refused, and a candidate this close to an observation the image already
    /// has is refused. `0` disables it.
    pub min_keypoint_separation_px: f64,
    /// The localizer's grid, window, sampler, search radius (`search`,
    /// patch-grid px), grazing cutoff (`min_grazing_cos`) and member
    /// localizability gate (`max_member_keypoint_uncertainty`, `0` disables
    /// it). Its per-view drop gates are not read.
    pub localize: KeypointLocalizeParams,
    /// The sub-pixel solve's step and convergence settings; its grid, window,
    /// sampler and robust iterations are taken from [`Self::localize`].
    pub refine: KeypointSubpixelParams,
}

/// The defaults are the rule and gate the leave-one-image-out evaluation chose
/// (see the spec, "Why the default is what it is"): a candidate passes when it
/// reaches either the image's pooled bar (the median minus three scaled
/// deviations of every candidate's references' leave-one-out ZNCCs) or its own
/// track's bar (0.9 of its references' median, or the pair rule at 0.9 for two
/// references); and its keypoint must lie within the median plus three scaled
/// deviations of the accepted keypoints' distances from their projections,
/// never less than one pixel. Every bar is read off the call's own data.
impl Default for AddImageToTracksOptions {
    fn default() -> Self {
        Self {
            rule: AcceptRule::PooledOrTrack {
                pooled: BasisStatistic::MedianMinusMad { k: 3.0 },
                track: BasisStatistic::FractionOfMedian { fraction: 0.9 },
                pair: PairRule {
                    statistic: PairStatistic::Mean,
                    factor: 0.9,
                },
            },
            min_zncc: 0.5,
            position_gate: PositionGate::ImageMad {
                k: 3.0,
                floor_px: 1.0,
            },
            template: TemplateSource::Rendered,
            require_facing: true,
            subpixel: true,
            ascend_on_edge: false,
            min_keypoint_separation_px: 1.0,
            localize: KeypointLocalizeParams::default(),
            refine: KeypointSubpixelParams::default(),
        }
    }
}

/// Why a candidate point was not added.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Refusal {
    /// The point carries no patch frame.
    NoPatch,
    /// The point projects behind the camera or outside the frame.
    NotInFrame,
    /// The new view's ray is near-parallel to the patch plane.
    Grazing,
    /// The new camera is on the other side of the patch plane from the cameras
    /// that observe the point.
    BackFacing,
    /// Fewer than two existing observations render in frame in a decoded image.
    TooFewReferences,
    /// The new view's own core pins no 2D position.
    Unlocalizable,
    /// No shift of the search window could be scored in frame.
    NoPeak,
    /// The correlation peak is on the edge of the searched window.
    PeakAtEdge,
    /// The final keypoint's core could not be rendered in frame.
    Unscorable,
    /// The ZNCC is below [`AddImageToTracksOptions::min_zncc`].
    BelowFloor,
    /// The rule's bar was not reached.
    BelowBar,
    /// The keypoint is further from the projection than the positional gate
    /// allows.
    TooFar,
    /// Another keypoint of the image is at the same place.
    SharedKeypoint,
}

impl Refusal {
    /// The snake-case name a report and the binding spell it with.
    pub fn name(&self) -> &'static str {
        match self {
            Refusal::NoPatch => "no_patch",
            Refusal::NotInFrame => "not_in_frame",
            Refusal::Grazing => "grazing",
            Refusal::BackFacing => "back_facing",
            Refusal::TooFewReferences => "too_few_references",
            Refusal::Unlocalizable => "unlocalizable",
            Refusal::NoPeak => "no_peak",
            Refusal::PeakAtEdge => "peak_at_edge",
            Refusal::Unscorable => "unscorable",
            Refusal::BelowFloor => "below_floor",
            Refusal::BelowBar => "below_bar",
            Refusal::TooFar => "too_far",
            Refusal::SharedKeypoint => "shared_keypoint",
        }
    }
}

/// What happened to one point the image did not observe.
#[derive(Debug, Clone, PartialEq)]
pub struct CandidateReport {
    /// The point's index, the same in the value that comes back.
    pub point: u32,
    /// `None` when the observation was added; the reason otherwise.
    pub refusal: Option<Refusal>,
    /// The point's projection into the image, source px.
    pub projection: Option<[f64; 2]>,
    /// The images of the references that rendered.
    pub references: Vec<u32>,
    /// Per reference, its ZNCC against the consensus of the others.
    pub reference_loo_zncc: Vec<f64>,
    /// The references' pairwise ZNCCs, row-major `n × n`.
    pub reference_pair_zncc: Vec<f64>,
    /// The keypoint the search found, before any sub-pixel step.
    pub search_keypoint: Option<[f64; 2]>,
    /// The final keypoint.
    pub keypoint: Option<[f64; 2]>,
    /// The final keypoint's distance from the projection, source px.
    pub offset_px: f64,
    /// The member localizability score of the new view's core at the
    /// projection, patch-grid px.
    pub sigma_pos: f64,
    /// The ZNCC at the search's integer peak.
    pub peak_zncc: f64,
    /// The new view's ZNCC against the consensus, at the final keypoint.
    pub zncc: f64,
    /// The new view's ZNCC against each reference, at the final keypoint.
    pub pair_zncc: Vec<f64>,
    /// The number the rule compared with [`Self::bar`]: the ZNCC, or for a
    /// two-reference track under [`AcceptRule::TrackBasis`] the pair
    /// statistic.
    pub judged: f64,
    /// The bar the rule set.
    pub bar: f64,
}

impl CandidateReport {
    fn new(point: u32) -> Self {
        Self {
            point,
            refusal: None,
            projection: None,
            references: Vec::new(),
            reference_loo_zncc: Vec::new(),
            reference_pair_zncc: Vec::new(),
            search_keypoint: None,
            keypoint: None,
            offset_px: f64::NAN,
            sigma_pos: f64::NAN,
            peak_zncc: f64::NAN,
            zncc: f64::NAN,
            pair_zncc: Vec::new(),
            judged: f64::NAN,
            bar: f64::NAN,
        }
    }
}

/// What one call did.
#[derive(Debug, Clone, PartialEq)]
pub struct AddImageToTracksReport {
    /// The image.
    pub image: usize,
    /// One entry per point the image did not observe, in point order.
    pub candidates: Vec<CandidateReport>,
    /// How many observations were added.
    pub accepted: usize,
    /// Observations before the call.
    pub observations_before: usize,
    /// Observations after it.
    pub observations_after: usize,
    /// The bar [`AcceptRule::PooledBasis`] set, when it was the rule.
    pub pooled_bar: Option<f64>,
    /// The bound [`PositionGate`] applied, in source pixels, when there was
    /// one.
    pub position_bound_px: Option<f64>,
}

impl AddImageToTracksReport {
    /// How many candidates each refusal took, in the order of first
    /// appearance.
    pub fn refusal_counts(&self) -> Vec<(Refusal, usize)> {
        let mut out: Vec<(Refusal, usize)> = Vec::new();
        for c in &self.candidates {
            if let Some(r) = c.refusal {
                match out.iter_mut().find(|(k, _)| *k == r) {
                    Some(slot) => slot.1 += 1,
                    None => out.push((r, 1)),
                }
            }
        }
        out
    }
}

/// Why a call produced nothing.
#[derive(Debug, Clone, PartialEq)]
pub enum AddImageToTracksError {
    /// The image index is not one of the value's images.
    NoSuchImage {
        /// The index asked for.
        image: usize,
        /// How many images the value has.
        image_count: usize,
    },
    /// The image carries no pose.
    Unposed(usize),
    /// Fewer pyramids were supplied than the value has images.
    PyramidsMissing {
        /// How many were supplied.
        got: usize,
        /// How many the value has.
        expected: usize,
    },
    /// The image's own photograph was not supplied.
    NoTargetImage(usize),
    /// The value's observations are `.sift` feature indexes, and an added
    /// observation has no feature to name.
    NotEmbeddedPatches,
    /// The value carries no patch frames.
    NoPatchFrames,
    /// The caller asked the call to stop.
    Cancelled,
}

impl std::fmt::Display for AddImageToTracksError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoSuchImage { image, image_count } => write!(
                f,
                "image {image} is not one of this reconstruction's {image_count} images"
            ),
            Self::Unposed(image) => write!(f, "image {image} has no pose to project into"),
            Self::PyramidsMissing { got, expected } => write!(
                f,
                "{got} images were supplied and the reconstruction has {expected}"
            ),
            Self::NoTargetImage(image) => {
                write!(f, "image {image}'s own photograph was not supplied")
            }
            Self::NotEmbeddedPatches => write!(
                f,
                "this reconstruction's observations are .sift feature indexes, and an added \
                 observation has no feature to name; convert it to embedded patches first"
            ),
            Self::NoPatchFrames => write!(
                f,
                "adding an image to tracks needs a patch frame per point, and this \
                 reconstruction carries none"
            ),
            Self::Cancelled => write!(f, "the call was cancelled before it had an answer"),
        }
    }
}

impl std::error::Error for AddImageToTracksError {}

impl From<Cancelled> for AddImageToTracksError {
    fn from(_: Cancelled) -> Self {
        Self::Cancelled
    }
}

/// Add observations of `image` to the points of `recon` it does not observe,
/// where it sees them and its view agrees with theirs.
///
/// `pyramids` holds one entry per image of `recon`: the decoded photograph, or
/// `None` for one not to hand. The target's own must be there. An image
/// without one is left out of every reference set, so a point whose
/// observations are mostly in images not supplied is refused for too few
/// references rather than failing the call. Poses and cameras are read from
/// `recon`.
///
/// Every point the image does not observe is reported, with its outcome and
/// the numbers it was judged on (see [`CandidateReport`]). The accepted
/// observations are written with their keypoint and, where the value has the
/// column, their ZNCC in `observation_confidence` on the bench commit's byte
/// scale, raised to `1` because `0` means unmeasured. Nothing else changes:
/// every point keeps its index, position, frame, bitmap, colour, constraint and
/// stored error, and every existing observation is untouched.
///
/// `progress` names the measuring and the write-back and is how the call is
/// asked to stop; a stopped call returns [`AddImageToTracksError::Cancelled`].
///
/// # Example
///
/// ```no_run
/// use sfmtool_core::progress::Progress;
/// use sfmtool_core::reconstruction::add_image_to_tracks::{
///     add_image_to_tracks, AddImageToTracksOptions,
/// };
/// # fn run(
/// #     recon: &sfmtool_core::SfmrReconstruction,
/// #     pyramids: &[Option<&sfmtool_core::camera::remap::ImageU8Pyramid>],
/// # ) -> Result<(), Box<dyn std::error::Error>> {
/// let (next, report) = add_image_to_tracks(
///     recon,
///     3,
///     pyramids,
///     &AddImageToTracksOptions::default(),
///     &Progress::none(),
/// )?;
/// println!("{} of {} candidates added", report.accepted, report.candidates.len());
/// # let _ = next;
/// # Ok(())
/// # }
/// ```
///
/// # Errors
///
/// [`AddImageToTracksError`] states which precondition did not hold, or that
/// the call was cancelled.
pub fn add_image_to_tracks(
    recon: &SfmrReconstruction,
    image: usize,
    pyramids: &[Option<&ImageU8Pyramid>],
    options: &AddImageToTracksOptions,
    progress: &Progress<'_>,
) -> Result<(SfmrReconstruction, AddImageToTracksReport), AddImageToTracksError> {
    let table = &recon.image_table;
    let image_count = table.images.len();
    if image >= image_count {
        return Err(AddImageToTracksError::NoSuchImage { image, image_count });
    }
    if pyramids.len() < image_count {
        return Err(AddImageToTracksError::PyramidsMissing {
            got: pyramids.len(),
            expected: image_count,
        });
    }
    let target = &table.images[image];
    if !is_posed(&target.quaternion_wxyz, &target.translation_xyz) {
        return Err(AddImageToTracksError::Unposed(image));
    }
    if pyramids[image].is_none() {
        return Err(AddImageToTracksError::NoTargetImage(image));
    }
    let keypoints = match &recon.point_set.observations {
        ObservationSource::EmbeddedPatches { keypoints_xy, .. } => keypoints_xy,
        ObservationSource::SiftFiles { .. } => {
            return Err(AddImageToTracksError::NotEmbeddedPatches)
        }
    };
    let (Some(u_col), Some(v_col)) = (
        recon.point_set.patch_u_halfvec_xyz.as_ref(),
        recon.point_set.patch_v_halfvec_xyz.as_ref(),
    ) else {
        return Err(AddImageToTracksError::NoPatchFrames);
    };
    progress.check_cancel()?;

    // One grid for every kernel: the stored bitmaps' where they are the
    // template, so a bitmap row and a rendered core are the same samples.
    let bitmaps = match options.template {
        TemplateSource::StoredBitmap => recon.point_set.patch_bitmaps_y_x_rgba.as_deref(),
        TemplateSource::Rendered => None,
    };
    let mut localize = KeypointLocalizeParams {
        search_resolution_multiplier: 1.0,
        ..options.localize.clone()
    };
    if let Some(b) = bitmaps {
        localize.resolution = b.shape()[1] as u32;
    }
    let refine = KeypointSubpixelParams {
        resolution: localize.resolution,
        window: localize.window,
        sampler: localize.sampler,
        robust_iters: localize.robust_iters,
        ..options.refine.clone()
    };

    // Poses and cameras from the value, so a view can never disagree with it.
    let poses: Vec<RigidTransform> = table
        .images
        .iter()
        .map(|im| {
            let q = im.quaternion_wxyz;
            let t = im.translation_xyz;
            RigidTransform::from_wxyz_translation([q.w, q.i, q.j, q.k], [t.x, t.y, t.z])
        })
        .collect();
    let views: Vec<Option<ProjectedImage<'_>>> = (0..image_count)
        .map(|i| {
            let im = &table.images[i];
            let pyramid = pyramids[i]?;
            is_posed(&im.quaternion_wxyz, &im.translation_xyz).then(|| ProjectedImage {
                camera: &table.cameras[im.camera_index as usize],
                cam_from_world: &poses[i],
                pyramid,
            })
        })
        .collect();
    let centers: Vec<Point3<f64>> = poses
        .iter()
        .map(|p| p.inverse_translation_origin())
        .collect();

    let set = &recon.point_set;
    let candidates: Vec<u32> = (0..set.points.len() as u32)
        .filter(|&p| {
            !recon
                .observations_for_point(p as usize)
                .iter()
                .any(|o| o.image_index as usize == image)
        })
        .collect();

    // ---- Measure every candidate ----
    let measured = {
        let _phase = progress.phase("measure candidates");
        let done = AtomicUsize::new(0);
        let total = candidates.len();
        let step = (total / 100).max(1);
        let ctx = Context {
            recon,
            image,
            keypoints,
            u_col,
            v_col,
            bitmaps,
            views: &views,
            centers: &centers,
            localize: &localize,
            refine: &refine,
            options,
        };
        let out: Vec<CandidateReport> = candidates
            .par_iter()
            .map(|&p| {
                if progress.is_cancelled() {
                    return CandidateReport::new(p);
                }
                let report = ctx.measure(p);
                let n = done.fetch_add(1, Ordering::Relaxed) + 1;
                if n.is_multiple_of(step) || n == total {
                    progress.count(n as u64, Some(total as u64), "points");
                }
                report
            })
            .collect();
        progress.check_cancel()?;
        out
    };
    let mut candidates = measured;

    // ---- Judge ----
    let pooled_bar = match options.rule {
        AcceptRule::PooledBasis { statistic }
        | AcceptRule::PooledOrTrack {
            pooled: statistic, ..
        } => {
            let pool: Vec<f64> = candidates
                .iter()
                .filter(|c| c.refusal.is_none())
                .flat_map(|c| c.reference_loo_zncc.iter().copied())
                .collect();
            Some(statistic.bar(&pool))
        }
        _ => None,
    };
    for c in candidates.iter_mut().filter(|c| c.refusal.is_none()) {
        judge(c, options, pooled_bar);
    }

    // ---- Where the keypoints landed ----
    let position_bound_px = match options.position_gate {
        PositionGate::Off => None,
        PositionGate::MaxPx(px) => Some(px),
        PositionGate::ImageMad { k, floor_px } => {
            let mut offsets: Vec<f64> = candidates
                .iter()
                .filter(|c| c.refusal.is_none())
                .map(|c| c.offset_px)
                .collect();
            (offsets.len() >= 3).then(|| {
                let med = median_in_place(&mut offsets);
                let mut dev: Vec<f64> = offsets.iter().map(|o| (o - med).abs()).collect();
                let mad = median_in_place(&mut dev);
                (med + k * MAD_TO_SIGMA * mad).max(floor_px)
            })
        }
    };
    if let Some(bound) = position_bound_px {
        for c in candidates.iter_mut().filter(|c| c.refusal.is_none()) {
            if c.offset_px.is_nan() || c.offset_px > bound {
                c.refusal = Some(Refusal::TooFar);
            }
        }
    }
    if options.min_keypoint_separation_px > 0.0 {
        separate(recon, image, keypoints, &mut candidates, options);
    }

    // ---- Write ----
    let accepted: Vec<(u32, [f64; 2], f64)> = candidates
        .iter()
        .filter(|c| c.refusal.is_none())
        .map(|c| {
            (
                c.point,
                c.keypoint.expect("an accepted candidate has a keypoint"),
                c.zncc,
            )
        })
        .collect();
    let observations_before = set.tracks.len();
    let next = {
        let _phase = progress.phase("write back");
        insert_observations(recon, image as u32, &accepted)
    };
    let report = AddImageToTracksReport {
        image,
        accepted: accepted.len(),
        observations_before,
        observations_after: next.point_set.tracks.len(),
        candidates,
        pooled_bar,
        position_bound_px,
    };
    progress_info!(
        progress,
        "{} of {} candidate points added to image {}",
        report.accepted,
        report.candidates.len(),
        image
    );
    Ok((next, report))
}

/// What measuring one candidate reads.
struct Context<'a, 'v> {
    recon: &'a SfmrReconstruction,
    image: usize,
    keypoints: &'a Array2<f32>,
    u_col: &'a Array2<f32>,
    v_col: &'a Array2<f32>,
    bitmaps: Option<&'a ndarray::Array4<u8>>,
    views: &'a [Option<ProjectedImage<'v>>],
    centers: &'a [Point3<f64>],
    localize: &'a KeypointLocalizeParams,
    refine: &'a KeypointSubpixelParams,
    options: &'a AddImageToTracksOptions,
}

impl Context<'_, '_> {
    /// Everything up to the verdict for point `p`: the visibility gates, the
    /// reference consensus, the search, the sub-pixel step and the score.
    fn measure(&self, p: u32) -> CandidateReport {
        let mut out = CandidateReport::new(p);
        let pi = p as usize;
        let point = &self.recon.point_set.points[pi];
        let u = Vector3::new(
            f64::from(self.u_col[[pi, 0]]),
            f64::from(self.u_col[[pi, 1]]),
            f64::from(self.u_col[[pi, 2]]),
        );
        let v = Vector3::new(
            f64::from(self.v_col[[pi, 0]]),
            f64::from(self.v_col[[pi, 1]]),
            f64::from(self.v_col[[pi, 2]]),
        );
        let (hu, hv) = (u.norm(), v.norm());
        if !(hu > 0.0 && hv > 0.0) {
            out.refusal = Some(Refusal::NoPatch);
            return out;
        }
        let mut patch = OrientedPatch::new(point.position, u / hu, v / hv, [hu, hv]);
        patch.w = if point.is_at_infinity() { 0.0 } else { 1.0 };

        let target = self.views[self.image].expect("the target's view was checked");
        let in_frame = |(x, y): (f64, f64)| {
            (x >= 0.0
                && y >= 0.0
                && x < f64::from(target.camera.width)
                && y < f64::from(target.camera.height))
            .then_some([x, y])
        };
        let Some(proj) = project_unclipped(&target, &patch.center, patch.w).and_then(in_frame)
        else {
            out.refusal = Some(Refusal::NotInFrame);
            return out;
        };
        out.projection = Some(proj);

        // Grazing and facing, on the plane through the point.
        let normal = patch.normal();
        let observations = self.recon.observations_for_point(pi);
        if patch.w != 0.0 {
            let c_t = self.centers[self.image];
            let d = patch.center - c_t;
            let cos = if d.norm() > 1e-12 {
                (d.dot(&normal) / d.norm()).abs()
            } else {
                0.0
            };
            if cos < self.localize.min_grazing_cos {
                out.refusal = Some(Refusal::Grazing);
                return out;
            }
            if self.options.require_facing {
                let side = |c: &Point3<f64>| (c - patch.center).dot(&normal).signum();
                let vote: f64 = observations
                    .iter()
                    .map(|o| side(&self.centers[o.image_index as usize]))
                    .sum();
                if vote != 0.0 && side(&c_t) != vote.signum() {
                    out.refusal = Some(Refusal::BackFacing);
                    return out;
                }
            }
        }

        // The references: every existing observation in a decoded, posed image.
        // Local view 0 is the target; the references follow.
        let start = self.recon.point_set.observation_offsets[pi];
        let mut local: Vec<ProjectedImage<'_>> = vec![target];
        let mut local_images: Vec<u32> = Vec::new();
        let mut ref_keypoints: Vec<[f64; 2]> = Vec::new();
        for (k, o) in observations.iter().enumerate() {
            if let Some(view) = self.views[o.image_index as usize] {
                local.push(view);
                local_images.push(o.image_index);
                let row = start + k;
                ref_keypoints.push([
                    f64::from(self.keypoints[[row, 0]]),
                    f64::from(self.keypoints[[row, 1]]),
                ]);
            }
        }
        let local_refs: Vec<u32> = (1..local.len() as u32).collect();
        let Some(mut consensus) =
            ReferenceConsensus::build(&patch, &local, &local_refs, &ref_keypoints, self.localize)
        else {
            out.refusal = Some(Refusal::TooFewReferences);
            return out;
        };
        out.references = consensus
            .references
            .iter()
            .map(|&l| local_images[l as usize - 1])
            .collect();
        out.reference_loo_zncc = consensus.loo_zncc.clone();
        out.reference_pair_zncc = consensus.pair_zncc.clone();
        if let Some(bitmaps) = self.bitmaps {
            let row = bitmaps.index_axis(ndarray::Axis(0), pi);
            if let Some(slice) = row.as_slice() {
                // A row with no texture (a point the file stores no bitmap
                // for) leaves the rendered consensus in place.
                consensus.use_bitmap_template(slice, bitmaps.shape()[3]);
            }
        }

        // The search, from the projection.
        let search = match consensus.search(
            &patch,
            &target,
            None,
            self.options.ascend_on_edge,
            self.localize,
        ) {
            Ok(s) => s,
            Err(_) => {
                out.refusal = Some(Refusal::NoPeak);
                return out;
            }
        };
        out.sigma_pos = search.sigma_pos;
        out.peak_zncc = search.peak_zncc;
        out.search_keypoint = search.keypoint;
        let tau = self.localize.max_member_keypoint_uncertainty;
        if tau.is_finite() && tau > 0.0 && search.sigma_pos > tau {
            out.refusal = Some(Refusal::Unlocalizable);
            return out;
        }
        let Some(mut keypoint) = search.keypoint else {
            out.refusal = Some(Refusal::NoPeak);
            return out;
        };
        if search.at_edge {
            out.refusal = Some(Refusal::PeakAtEdge);
            return out;
        }
        if self.options.subpixel {
            let refs_used: Vec<u32> = consensus.references.clone();
            let kps_used: Vec<[f64; 2]> = refs_used
                .iter()
                .map(|&l| ref_keypoints[l as usize - 1])
                .collect();
            if let Some(refined) = refine_view_against_references(
                &patch,
                &local,
                &refs_used,
                &kps_used,
                0,
                keypoint,
                self.refine,
            ) {
                keypoint = refined;
            }
        }
        out.keypoint = Some(keypoint);
        out.offset_px = (keypoint[0] - proj[0]).hypot(keypoint[1] - proj[1]);
        let Some(score) = consensus.score(&patch, &target, keypoint, self.localize) else {
            out.refusal = Some(Refusal::Unscorable);
            return out;
        };
        out.zncc = score.zncc;
        out.pair_zncc = score.pair_zncc;
        out
    }
}

/// Apply the photometric rule to a measured candidate.
fn judge(c: &mut CandidateReport, options: &AddImageToTracksOptions, pooled_bar: Option<f64>) {
    let floor = options.min_zncc;
    let (judged, bar) = match options.rule {
        AcceptRule::FixedZncc => (c.zncc, floor),
        AcceptRule::TrackBasis { statistic, pair } => track_judgement(c, statistic, pair),
        AcceptRule::PooledBasis { .. } => (c.zncc, pooled_bar.unwrap_or(f64::NAN)),
        AcceptRule::PooledOrTrack { track, pair, .. } => {
            let pooled = pooled_bar.unwrap_or(f64::NAN);
            if c.zncc >= pooled {
                (c.zncc, pooled)
            } else {
                track_judgement(c, track, pair)
            }
        }
    };
    c.judged = judged;
    c.bar = bar;
    let floor_applies = floor > 0.0 && !matches!(options.rule, AcceptRule::FixedZncc);
    if floor_applies && (c.zncc.is_nan() || c.zncc < floor) {
        c.refusal = Some(Refusal::BelowFloor);
    } else if judged.is_nan() || bar.is_nan() || judged < bar {
        c.refusal = Some(Refusal::BelowBar);
    }
}

/// The number a track's own references judge and the bar they set: a
/// statistic of their leave-one-out ZNCCs with three or more, the pair rule
/// with two.
fn track_judgement(c: &CandidateReport, statistic: BasisStatistic, pair: PairRule) -> (f64, f64) {
    if c.references.len() >= 3 {
        return (c.zncc, statistic.bar(&c.reference_loo_zncc));
    }
    let values = &c.pair_zncc;
    let judged = match pair.statistic {
        PairStatistic::Min => values.iter().copied().fold(f64::INFINITY, f64::min),
        PairStatistic::Mean => values.iter().sum::<f64>() / values.len() as f64,
        PairStatistic::Max => values.iter().copied().fold(f64::NEG_INFINITY, f64::max),
    };
    let between = c.reference_pair_zncc.get(1).copied().unwrap_or(f64::NAN);
    (judged, pair.factor * between)
}

/// One observation per place in the image: refuse an accepted candidate that
/// lands on an observation the image already has, and of two accepted
/// candidates that land together keep the one with the higher ZNCC.
fn separate(
    recon: &SfmrReconstruction,
    image: usize,
    keypoints: &Array2<f32>,
    candidates: &mut [CandidateReport],
    options: &AddImageToTracksOptions,
) {
    let sep = options.min_keypoint_separation_px;
    let cell = |xy: [f64; 2]| ((xy[0] / sep).floor() as i64, (xy[1] / sep).floor() as i64);
    let mut grid: HashMap<(i64, i64), Vec<[f64; 2]>> = HashMap::new();
    for (row, o) in recon.point_set.tracks.iter().enumerate() {
        if o.image_index as usize == image {
            let xy = [
                f64::from(keypoints[[row, 0]]),
                f64::from(keypoints[[row, 1]]),
            ];
            grid.entry(cell(xy)).or_default().push(xy);
        }
    }
    let mut order: Vec<usize> = (0..candidates.len())
        .filter(|&i| candidates[i].refusal.is_none())
        .collect();
    order.sort_by(|&a, &b| {
        candidates[b]
            .zncc
            .partial_cmp(&candidates[a].zncc)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    for i in order {
        let xy = candidates[i]
            .keypoint
            .expect("an accepted candidate has a keypoint");
        let (cx, cy) = cell(xy);
        let taken = (-1..=1).any(|dx| {
            (-1..=1).any(|dy| {
                grid.get(&(cx + dx, cy + dy))
                    .is_some_and(|pts| pts.iter().any(|q| (q[0] - xy[0]).hypot(q[1] - xy[1]) < sep))
            })
        });
        if taken {
            candidates[i].refusal = Some(Refusal::SharedKeypoint);
        } else {
            grid.entry((cx, cy)).or_default().push(xy);
        }
    }
}

/// `source` with one observation of `image` added to each point in `accepted`
/// (`(point, keypoint, zncc)`, each point at most once), in image order within
/// its track. Every other column travels verbatim.
fn insert_observations(
    source: &SfmrReconstruction,
    image: u32,
    accepted: &[(u32, [f64; 2], f64)],
) -> SfmrReconstruction {
    if accepted.is_empty() {
        return source.clone();
    }
    let set = &source.point_set;
    let mut added: HashMap<u32, ([f64; 2], f64)> = HashMap::with_capacity(accepted.len());
    for &(p, kp, z) in accepted {
        added.insert(p, (kp, z));
    }
    let ObservationSource::EmbeddedPatches {
        keypoints_xy,
        image_file_hashes,
    } = &set.observations
    else {
        unreachable!("the caller refused a sift_files value");
    };
    let total = set.tracks.len() + accepted.len();
    let mut tracks: Vec<TrackObservation> = Vec::with_capacity(total);
    let mut kp_flat: Vec<f32> = Vec::with_capacity(2 * total);
    let mut confidence: Option<Vec<u8>> = set
        .observation_confidence
        .as_ref()
        .map(|_| Vec::with_capacity(total));
    let mut counts = set.observation_counts.clone();
    for (p, count) in counts.iter_mut().enumerate() {
        let start = set.observation_offsets[p];
        let end = set.observation_offsets[p + 1];
        let mut pending = added.get(&(p as u32)).copied();
        let push_new = |tracks: &mut Vec<TrackObservation>,
                        kp_flat: &mut Vec<f32>,
                        confidence: &mut Option<Vec<u8>>,
                        (kp, z): ([f64; 2], f64)| {
            tracks.push(TrackObservation {
                image_index: image,
                point_index: p as u32,
            });
            kp_flat.push(kp[0] as f32);
            kp_flat.push(kp[1] as f32);
            if let Some(c) = confidence.as_mut() {
                c.push(confidence_byte(z));
            }
        };
        for row in start..end {
            if let Some(new) = pending {
                if set.tracks[row].image_index > image {
                    push_new(&mut tracks, &mut kp_flat, &mut confidence, new);
                    pending = None;
                }
            }
            tracks.push(set.tracks[row]);
            kp_flat.push(keypoints_xy[[row, 0]]);
            kp_flat.push(keypoints_xy[[row, 1]]);
            if let (Some(c), Some(old)) = (confidence.as_mut(), set.observation_confidence.as_ref())
            {
                c.push(old[row]);
            }
        }
        if let Some(new) = pending {
            push_new(&mut tracks, &mut kp_flat, &mut confidence, new);
        }
        if added.contains_key(&(p as u32)) {
            *count += 1;
        }
    }
    let n = tracks.len();
    let mut next = source.clone();
    let out = &mut next.point_set;
    out.tracks = tracks;
    out.observation_counts = counts;
    out.observations = ObservationSource::EmbeddedPatches {
        keypoints_xy: Array2::from_shape_vec((n, 2), kp_flat)
            .expect("two coordinates per observation"),
        image_file_hashes: image_file_hashes.clone(),
    };
    out.observation_confidence = confidence;
    out.rebuild_derived_fields(source.image_count());
    next.metadata.observation_count = n as u32;
    // The hashes on a value describe the file it came from, and this value is
    // not that file.
    next.content_hash = ContentHash::default();
    next
}

/// A ZNCC on the bench commit's byte scale, never the `0` that means
/// unmeasured.
fn confidence_byte(zncc: f64) -> u8 {
    if zncc.is_finite() {
        ((zncc.clamp(0.0, 1.0) * f64::from(u8::MAX)).round() as u8).max(1)
    } else {
        1
    }
}

#[cfg(test)]
mod tests;
