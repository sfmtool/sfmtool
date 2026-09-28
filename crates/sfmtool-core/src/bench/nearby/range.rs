// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Distance ranges: how far along a pixel's ray a set of sightings allows the
//! point to be, and what that range says about the point.
//!
//! `specs/core/bench/distance-range.md` is the design. A range is found by
//! moving a point along the ray and watching how far each other sighting's
//! projection strays from where the sighting put it; the range ends where the
//! worst of them leaves its tolerance.

use nalgebra::Vector3;

use crate::bench::track_at_pixel::ViewCamera;
use crate::patch::normal_refine::ProjectedImage;

/// Where the search for the near end starts when the sightings put the point
/// at infinity: far enough out that no capture this tool reconstructs tells
/// it apart from infinity, so halving from here finds the near end.
const START_AT_INFINITY: f64 = 1e7;

/// How many times the search doubles or halves the distance before giving up
/// on finding an end. `2^40` covers any distance from the start that a finite
/// `f64` reconstruction holds.
const MAX_STEPS: usize = 40;

/// How many bisection steps in log distance refine an end once it is
/// bracketed: `2^-10` of a factor of two, about 0.07 %.
const BISECTION_STEPS: usize = 10;

/// The harness's range parameters, which a caller of [`distance_range`] and
/// [`classify_range`] passes on.
///
/// Kept together because the finder of nearby tracks takes them as one group of
/// its options; the defaults are the track-at-pixel harness's `range_px`,
/// `max_span` and `far_spread`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RangeOptions {
    /// The reprojection error, in px, each other sighting may reach inside the
    /// range, before [`distance_range`] widens it to the error at the point's
    /// own distance.
    pub tolerance_px: f64,
    /// The widest a bounded range may be, as its far end over its near end.
    pub max_span: f64,
    /// A range with no far end is far when its near end is at least this many
    /// times the [`camera_spread`].
    pub far_spread: f64,
}

impl Default for RangeOptions {
    fn default() -> Self {
        Self {
            tolerance_px: 1.0,
            max_span: 3.0,
            far_spread: 5.0,
        }
    }
}

/// What a range says about a point: whether the photographs pin its distance
/// down, or put it too far out to tell apart from infinity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct RangeClass {
    /// Finite at both ends, with a near end above zero, and no wider than
    /// [`RangeOptions::max_span`], far end over near end.
    pub bounded: bool,
    /// No far end, and a near end at least [`RangeOptions::far_spread`] times
    /// the [`camera_spread`].
    pub far: bool,
}

impl RangeClass {
    /// Whether the range says anything a comparison between ranges can use:
    /// it is bounded or far.
    pub fn usable(self) -> bool {
        self.bounded || self.far
    }
}

/// Why a range could not be computed.
#[derive(Debug, Clone, PartialEq)]
pub enum DistanceRangeError {
    /// The queried image, or a sighting's, is not one of `views`.
    NoSuchImage {
        /// The image asked for.
        image: u32,
        /// How many images `views` holds.
        image_count: usize,
    },
}

impl std::fmt::Display for DistanceRangeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoSuchImage { image, image_count } => write!(
                f,
                "image {image} is not one of the reconstruction's {image_count} images"
            ),
        }
    }
}

impl std::error::Error for DistanceRangeError {}

/// The distances along `pixel`'s ray in `image` at which every other sighting
/// stays within its tolerance: `[near, far]`, with `far` infinite when no
/// distance is too far.
///
/// `sightings` are `(image, pixel)` pairs of the point, and `distance` is the
/// distance along the ray, from the queried camera's centre, it was
/// triangulated at: `f64::INFINITY` for a point at infinity. Sightings in
/// `image` itself are not checked, since every point on the ray lands on the
/// pixel there.
///
/// The error at a distance is the largest pixel distance between a sighting
/// and where the point at that distance projects in the sighting's image; a
/// sighting whose image cannot see the point counts as an infinite error. The
/// tolerance is `tolerance_px`, or half a pixel more than the error at
/// `distance` when that is larger, so sightings that only meet within 2 px
/// still give a range. Each end is found by doubling the distance away from
/// `distance` (from `1e7` when it is infinite) until the error leaves the
/// tolerance, then bisecting ten times in log distance. `near` is `0` when the
/// search reaches no distance too near; `far` is infinite when `distance` is,
/// or when the error at infinity is within the tolerance.
///
/// `views` holds one entry per image of the reconstruction; only their cameras
/// and poses are read.
pub fn distance_range(
    views: &[ProjectedImage<'_>],
    image: u32,
    pixel: [f64; 2],
    sightings: &[(u32, [f64; 2])],
    distance: f64,
    tolerance_px: f64,
) -> Result<[f64; 2], DistanceRangeError> {
    let image_count = views.len();
    let camera = |i: u32| {
        views
            .get(i as usize)
            .map(ViewCamera::new)
            .ok_or(DistanceRangeError::NoSuchImage {
                image: i,
                image_count,
            })
    };
    let cq = camera(image)?;
    let others = sightings
        .iter()
        .filter(|&&(i, _)| i != image)
        .map(|&(i, p)| Ok((camera(i)?, p)))
        .collect::<Result<Vec<_>, DistanceRangeError>>()?;
    let ray = cq.ray(pixel).normalize();
    let error = RayError {
        center: cq.center,
        ray,
        others: &others,
    };
    let err = |t: f64| error.at(t);

    let tol = tolerance_px.max(err(distance) + 0.5);
    let bisect = |mut inside: f64, mut outside: f64| {
        for _ in 0..BISECTION_STEPS {
            let mid = (inside * outside).sqrt();
            if err(mid) <= tol {
                inside = mid;
            } else {
                outside = mid;
            }
        }
        inside
    };

    let start = if distance.is_finite() {
        distance
    } else {
        START_AT_INFINITY
    };
    let mut near = 0.0;
    let mut t = start;
    for _ in 0..MAX_STEPS {
        if err(t / 2.0) > tol {
            near = bisect(t, t / 2.0);
            break;
        }
        t /= 2.0;
    }
    let mut far = f64::INFINITY;
    if !(distance.is_infinite() || err(f64::INFINITY) <= tol) {
        let mut t = distance;
        for _ in 0..MAX_STEPS {
            if err(t * 2.0) > tol {
                far = bisect(t, t * 2.0);
                break;
            }
            t *= 2.0;
        }
    }
    Ok([near, far])
}

/// The worst sighting error of a point moved along one ray.
struct RayError<'a, 'v> {
    center: Vector3<f64>,
    ray: Vector3<f64>,
    others: &'a [(ViewCamera<'v>, [f64; 2])],
}

impl RayError<'_, '_> {
    /// The largest distance, in px, between a sighting and where the point `t`
    /// along the ray lands in its image; infinite when an image cannot see it,
    /// zero when there is nothing to check.
    fn at(&self, t: f64) -> f64 {
        let mut worst = 0.0f64;
        let x = self.center + self.ray * t;
        for (camera, p) in self.others {
            let q = if t.is_infinite() {
                camera.project_direction(&self.ray)
            } else {
                camera.project_homogeneous(&x, 1.0)
            };
            let Some(q) = q else {
                return f64::INFINITY;
            };
            worst = worst.max((q[0] - p[0]).hypot(q[1] - p[1]));
        }
        worst
    }
}

/// The largest distance between two camera centres of `views`, the scale the
/// far classification and the far-field sweep measure distances against.
///
/// Zero for fewer than two views.
pub fn camera_spread(views: &[ProjectedImage<'_>]) -> f64 {
    let centers: Vec<Vector3<f64>> = views.iter().map(|v| ViewCamera::new(v).center).collect();
    let mut spread = 0.0f64;
    for (a, ca) in centers.iter().enumerate() {
        for cb in &centers[a + 1..] {
            spread = spread.max((ca - cb).norm());
        }
    }
    spread
}

/// Whether `range` is bounded, far, both or neither, with `camera_spread` the
/// reconstruction's [`camera_spread`].
///
/// Only [`RangeOptions::max_span`] and [`RangeOptions::far_spread`] are read.
pub fn classify_range(range: [f64; 2], camera_spread: f64, options: &RangeOptions) -> RangeClass {
    let [near, far] = range;
    RangeClass {
        bounded: near > 0.0 && far.is_finite() && far / near <= options.max_span,
        far: far.is_infinite() && near >= options.far_spread * camera_spread,
    }
}
