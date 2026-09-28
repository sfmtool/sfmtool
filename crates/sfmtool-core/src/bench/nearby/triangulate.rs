// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The least-squares meeting point of a set of sightings' rays, with each
//! sighting's reprojection error there, and the vetting that drops the worst
//! sighting while three or more remain.
//!
//! `specs/core/bench/nearby-sources.md` is the design.

use nalgebra::{Matrix3, Vector3};

use crate::bench::track_at_pixel::ViewCamera;
use crate::patch::normal_refine::ProjectedImage;

/// Sightings as `(image, pixel)` pairs, in the order a source found them.
pub(super) type Sightings = Vec<(u32, [f64; 2])>;

/// Where a set of sightings' rays meet, and how far each sighting is from
/// where that point projects.
#[derive(Debug, Clone, PartialEq)]
pub struct RayMeeting {
    /// The point, in world coordinates.
    pub position: Vector3<f64>,
    /// Each sighting's reprojection error at [`Self::position`], in px, in the
    /// order the sightings were given.
    pub errors_px: Vec<f64>,
}

impl RayMeeting {
    /// The largest of [`Self::errors_px`].
    pub fn max_error_px(&self) -> f64 {
        self.errors_px
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max)
    }
}

/// The point nearest every sighting's ray in the least-squares sense, and each
/// sighting's reprojection error there.
///
/// Each sighting `(image, pixel)` contributes the ray from its camera centre
/// through its pixel, and the point minimises the sum of squared distances to
/// the rays. `None` when the rays fix no point (they are all parallel), when a
/// sighting's image is not one of `views`, or when the point is behind a
/// sighting's camera or outside its lens model.
///
/// # Example
///
/// ```no_run
/// # use sfmtool_core::bench::triangulate_sightings;
/// # fn run(views: &[sfmtool_core::patch::normal_refine::ProjectedImage<'_>]) {
/// let sightings = [(0, [412.0, 230.5]), (3, [398.2, 241.0]), (5, [420.7, 228.9])];
/// if let Some(met) = triangulate_sightings(views, &sightings) {
///     println!("{:?} within {:.2} px", met.position, met.max_error_px());
/// }
/// # }
/// ```
pub fn triangulate_sightings(
    views: &[ProjectedImage<'_>],
    sightings: &[(u32, [f64; 2])],
) -> Option<RayMeeting> {
    let mut cameras: Vec<Option<ViewCamera<'_>>> = (0..views.len()).map(|_| None).collect();
    for &(image, _) in sightings {
        let view = views.get(image as usize)?;
        cameras[image as usize].get_or_insert_with(|| ViewCamera::new(view));
    }
    let lookup = |i: u32| cameras[i as usize].as_ref().expect("filled above");
    meet(lookup, sightings)
}

/// [`triangulate_sightings`] with the cameras already built, one per image.
pub(super) fn meet_rays(
    cameras: &[ViewCamera<'_>],
    sightings: &[(u32, [f64; 2])],
) -> Option<RayMeeting> {
    meet(|i| &cameras[i as usize], sightings)
}

fn meet<'c, 'v: 'c>(
    camera: impl Fn(u32) -> &'c ViewCamera<'v>,
    sightings: &[(u32, [f64; 2])],
) -> Option<RayMeeting> {
    let mut a = Matrix3::<f64>::zeros();
    let mut b = Vector3::<f64>::zeros();
    for &(image, pixel) in sightings {
        let cam = camera(image);
        let d = cam.ray(pixel);
        let p = Matrix3::identity() - d * d.transpose();
        a += p;
        b += p * cam.center;
    }
    let x = a.lu().solve(&b)?;
    let mut errors_px = Vec::with_capacity(sightings.len());
    for &(image, pixel) in sightings {
        let cam = camera(image);
        let q = cam.project_homogeneous(&x, 1.0)?;
        if cam.depth(&x) <= 0.0 {
            return None;
        }
        let (dx, dy) = (q[0] - pixel[0], q[1] - pixel[1]);
        errors_px.push((dx * dx + dy * dy).sqrt());
    }
    Some(RayMeeting {
        position: x,
        errors_px,
    })
}

/// Triangulate `sightings`, dropping the worst while three or more remain,
/// until every error is within `max_reproj_px`; never the sighting in
/// `image`, the queried one.
///
/// Returns the sightings kept, in their order, and where they meet; `None`
/// when the rays fix no point, when fewer than two remain, or when the worst
/// is the queried sighting.
pub(super) fn meet_dropping_worst(
    cameras: &[ViewCamera<'_>],
    image: u32,
    mut sightings: Sightings,
    max_reproj_px: f64,
) -> Option<(Sightings, RayMeeting)> {
    while sightings.len() >= 2 {
        let met = meet_rays(cameras, &sightings);
        if let Some(met) = &met {
            if met.max_error_px() <= max_reproj_px {
                return Some((sightings, met.clone()));
            }
        }
        let met = met?;
        if sightings.len() < 3 {
            return None;
        }
        // The first of the largest, as the harness's `argmax` picks it.
        let mut worst = 0;
        for (k, &e) in met.errors_px.iter().enumerate() {
            if e > met.errors_px[worst] {
                worst = k;
            }
        }
        if sightings[worst].0 == image {
            return None;
        }
        sightings.remove(worst);
    }
    None
}
