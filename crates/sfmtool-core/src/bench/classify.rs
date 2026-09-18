// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The one rule that decides whether a bench track's rays fix a point or only a
//! bearing.
//!
//! `specs/core/bench/editable-track.md` § "Finite points and bearings" is the
//! design. Every bench step that triangulates -- the track-stage
//! [`fit`](super::fit::fit) and the cluster-to-track upgrade
//! ([`set_stage`](super::stage::set_stage())) -- ends at
//! [`classify_track_rays`], so the two cannot come to disagree about which
//! representation one set of sightings has earned.
//!
//! The criterion is not the bench's own. It is
//! [`classify_rays_at_infinity`], the same per-track test
//! [`classify_points_at_infinity`](crate::SfmrReconstruction::classify_points_at_infinity)
//! reclassifies a whole reconstruction with and
//! [`find_points_at_infinity`](crate::SfmrReconstruction::find_points_at_infinity)
//! admits discovered tracks on, called here over one track's rays with the same
//! defaults. What the bench adds is the third answer's disposal: a
//! reconstruction pass leaves an *indeterminate* track alone, because leaving it
//! alone is an option when the pass is relabel-only, and a fit has to write
//! something. A track whose baseline cannot resolve a scene-scale depth is
//! written as the bearing it is.

use std::fmt;

use nalgebra::{Point3, Vector3};

use crate::analysis::infinity::{
    camera_extents, classify_rays_at_infinity, Classification, DEFAULT_INVERSE_DEPTH_Z_CUTOFF,
    DEFAULT_NOISE_FLOOR_PX,
};
use crate::patch::normal_refine::ProjectedImage;

/// The measurement noise a bench classification assumes at each sighting, in
/// source-image px.
///
/// The reconstruction's own floor, unchanged: the per-ray angular noise is this
/// over the observing camera's focal length, which is what turns a spread of
/// rays into a depth uncertainty. A bench track's sightings are localized by the
/// same kernels the embed pass runs, so they carry the same floor.
pub const DEFAULT_CLASSIFY_NOISE_FLOOR_PX: f64 = DEFAULT_NOISE_FLOOR_PX;

/// The inverse-depth z-score a track's depth has to reach to be written as a
/// finite point.
///
/// [`DEFAULT_INVERSE_DEPTH_Z_CUTOFF`], the reconstruction's own bar.
pub const DEFAULT_CLASSIFY_Z_CUTOFF: f64 = DEFAULT_INVERSE_DEPTH_Z_CUTOFF;

/// Which of the criterion's three tests settled a track, and so what the
/// sentence beside the verdict says.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClassificationReason {
    /// The solve is conditioned well enough that the depth is observable without
    /// a noise model at all, which is the criterion's cheap pre-filter.
    WellConditioned,
    /// The depth stands clear of its own uncertainty: the inverse-depth z-score
    /// reached the bar.
    DepthResolved,
    /// The depth does not stand clear of its own uncertainty: the z-score fell
    /// short of the bar, or the solve was degenerate or behind a camera.
    DepthUnresolved,
    /// The observing baseline cannot place a point even at the capture's own
    /// scale, so neither answer is earned on the numbers and the honest one is
    /// the bearing.
    BaselineTooShort,
}

/// What one set of a track's rays resolves to, and the numbers behind the call.
///
/// [`Self::coordinate`] is what the track stores: a world point when
/// [`Self::at_infinity`] is false, and the unit bearing direction when it is
/// true. That is the `.sfmr` rule for a `w = 0` row, so a caller writes this
/// straight into a position and a `w` without asking which it has.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrackClassification {
    /// Whether the rays fix a bearing only, which is `w == 0`.
    pub at_infinity: bool,
    /// The coordinate the track takes: the triangulated point when finite, the
    /// unit direction when at infinity.
    pub coordinate: Point3<f64>,
    /// Which test settled it.
    pub reason: ClassificationReason,
    /// The triangulation's condition number.
    pub condition_number: f64,
    /// The inverse-depth z-score, `NaN` where the pre-filter settled the call
    /// before the noise model was consulted.
    pub inverse_depth_z: f64,
    /// The bar that z-score was judged against.
    pub inverse_depth_z_cutoff: f64,
    /// The farthest depth this track's geometry can tell from infinity, in world
    /// units. `NaN` where the pre-filter settled the call.
    pub resolvable_distance: f64,
    /// The distance the resolvable one had to reach: the camera cloud's extent,
    /// which is the scale of the region the capture explored.
    pub finite_horizon: f64,
    /// The widest angle between any two of the rays, in degrees. Not a test --
    /// it is the number a person reads a near-parallel track by, and it is the
    /// one the Point Track Detail panel already shows for a committed track.
    pub max_pair_angle_deg: f64,
}

impl fmt::Display for TrackClassification {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let c = self.coordinate;
        if self.at_infinity {
            write!(f, "at infinity along ({:.4}, {:.4}, {:.4})", c.x, c.y, c.z)?;
        } else {
            write!(f, "finite at ({:.4}, {:.4}, {:.4})", c.x, c.y, c.z)?;
        }
        write!(f, ": ")?;
        match self.reason {
            ClassificationReason::WellConditioned => write!(
                f,
                "condition number {:.0} under the {:.0} bar",
                self.condition_number,
                crate::analysis::infinity::CONDITION_NUMBER_PREFILTER
            )?,
            ClassificationReason::DepthResolved => write!(
                f,
                "inverse-depth z {:.2} over the {:.2} bar",
                self.inverse_depth_z, self.inverse_depth_z_cutoff
            )?,
            ClassificationReason::DepthUnresolved => write!(
                f,
                "inverse-depth z {:.2} under the {:.2} bar",
                self.inverse_depth_z, self.inverse_depth_z_cutoff
            )?,
            ClassificationReason::BaselineTooShort => write!(
                f,
                "the baseline tells depth apart from infinity only to {:.3}, short of \
                 the {:.3} the capture spans",
                self.resolvable_distance, self.finite_horizon
            )?,
        }
        write!(f, ", rays up to {:.3} deg apart", self.max_pair_angle_deg)
    }
}

/// One track's rays in world, and the lens each was measured through.
///
/// Built once per triangulation and handed to both the solve and the
/// classification, so the two read one set of rays.
pub struct TrackRays {
    /// Unit world-space ray per sighting.
    pub dirs: Vec<Vector3<f64>>,
    /// The camera centre each was cast from.
    pub centers: Vec<Point3<f64>>,
    /// The largest focal length of the camera each was measured through, in px,
    /// which is what turns the pixel noise floor into an angular one.
    pub focal_max: Vec<f64>,
}

impl TrackRays {
    /// How many rays there are.
    pub fn len(&self) -> usize {
        self.dirs.len()
    }

    /// Whether there are none.
    pub fn is_empty(&self) -> bool {
        self.dirs.is_empty()
    }
}

/// Decide whether `rays` fix a point or only a bearing, on the reconstruction's
/// own criterion.
///
/// `noise_floor_px` is the per-sighting measurement noise and `z_cutoff` the
/// inverse-depth z-score a depth has to reach; both default to the constants at
/// the top of this module, which are the reconstruction's. `images` supplies the
/// camera cloud the `finite_horizon` is measured over: the extent of **every**
/// image's centre, not just the observing ones, because the horizon is the scale
/// of the region the capture explored and a track observed by three neighbouring
/// frames is still a track in that capture.
///
/// # Example
///
/// ```no_run
/// # use sfmtool_core::bench::classify::{classify_track_rays, DEFAULT_CLASSIFY_NOISE_FLOOR_PX,
/// #     DEFAULT_CLASSIFY_Z_CUTOFF};
/// # fn run(
/// #     rays: &sfmtool_core::bench::classify::TrackRays,
/// #     images: &[sfmtool_core::patch::normal_refine::ProjectedImage<'_>],
/// # ) {
/// let call = classify_track_rays(
///     rays,
///     images,
///     DEFAULT_CLASSIFY_NOISE_FLOOR_PX,
///     DEFAULT_CLASSIFY_Z_CUTOFF,
/// );
/// println!("{call}");   // "at infinity along (…): inverse-depth z 2.41 under the 4.00 bar, …"
/// # }
/// ```
pub fn classify_track_rays(
    rays: &TrackRays,
    images: &[ProjectedImage<'_>],
    noise_floor_px: f64,
    z_cutoff: f64,
) -> TrackClassification {
    let centers: Vec<Point3<f64>> = images
        .iter()
        .map(|view| view.cam_from_world.inverse_translation_origin())
        .collect();
    let finite_horizon = camera_extents(&centers);
    let sigma_rad: Vec<f64> = rays
        .focal_max
        .iter()
        .map(|&f| if f > 0.0 { noise_floor_px / f } else { 0.0 })
        .collect();
    let rc = classify_rays_at_infinity(
        &rays.dirs,
        &rays.centers,
        &sigma_rad,
        z_cutoff,
        finite_horizon,
    );
    // The criterion leaves the noise-calibrated diagnostics `NaN` when its cheap
    // condition-number pre-filter settled the call, which is exactly how a
    // caller tells the two finite paths apart.
    let (at_infinity, coordinate, reason) = match rc.class {
        Classification::Finite(point) => (
            false,
            point,
            if rc.inverse_depth_z.is_nan() {
                ClassificationReason::WellConditioned
            } else {
                ClassificationReason::DepthResolved
            },
        ),
        Classification::Infinity(direction) => {
            (true, direction, ClassificationReason::DepthUnresolved)
        }
        // A reconstruction pass can leave an indeterminate track alone; a fit has
        // to write something, and what the numbers say is that this track's
        // depth is not observable.
        Classification::Indeterminate => (true, rc.bearing, ClassificationReason::BaselineTooShort),
    };
    TrackClassification {
        at_infinity,
        coordinate,
        reason,
        condition_number: rc.condition_number,
        inverse_depth_z: rc.inverse_depth_z,
        inverse_depth_z_cutoff: z_cutoff,
        resolvable_distance: rc.resolvable_distance,
        finite_horizon,
        max_pair_angle_deg: max_pair_angle_deg(&rays.dirs),
    }
}

/// The widest angle between any two of `dirs`, in degrees.
///
/// Quadratic in the track length, which is what a bench track is: tens of
/// sightings, once per fit.
fn max_pair_angle_deg(dirs: &[Vector3<f64>]) -> f64 {
    let mut widest = 0.0_f64;
    for (i, a) in dirs.iter().enumerate() {
        for b in &dirs[i + 1..] {
            let cos = a.dot(b).clamp(-1.0, 1.0);
            widest = widest.max(cos.acos().to_degrees());
        }
    }
    widest
}
