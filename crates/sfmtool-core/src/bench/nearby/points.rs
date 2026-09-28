// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The reconstruction's own points near a pixel, as candidate tracks.
//!
//! `specs/core/bench/nearby-sources.md` is the design.

use crate::bench::track_at_pixel::{ObservationIndex, ViewCamera};
use crate::patch::normal_refine::ProjectedImage;
use crate::reconstruction::edited::EditedReconstruction;

use super::candidate::{candidate, check_query, NearbyCandidate, NearbySource, NearbySourceError};

/// What [`nearby_points`] keeps. The defaults are the harness's.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PointsOptions {
    /// How far from the pixel a point's observation in the queried image may
    /// be, in px (harness `track_radius_px`).
    pub radius_px: f64,
    /// The most points returned, the nearest (harness `track_max`).
    pub max_points: usize,
    /// The fewest observations a point needs (harness `track_min_views`).
    pub min_views: usize,
    /// The largest reprojection error any observation may have, in px
    /// (harness `max_reproj_px`).
    pub max_reproj_px: f64,
}

impl Default for PointsOptions {
    fn default() -> Self {
        Self {
            radius_px: 40.0,
            max_points: 8,
            min_views: 2,
            max_reproj_px: 2.0,
        }
    }
}

/// The reconstruction's points observed within [`PointsOptions::radius_px`]
/// of `pixel` in `image`, nearest first, as candidate tracks.
///
/// A point is kept when it is finite, has at least
/// [`PointsOptions::min_views`] observations, and every observation lies
/// within [`PointsOptions::max_reproj_px`] of where the point projects in its
/// image; at most [`PointsOptions::max_points`] are kept. Each candidate's
/// sightings are the point's observations, its observation in `image` first
/// and the rest in track order, and it names the point in
/// [`NearbyCandidate::point`] and [`NearbyCandidate::id`].
///
/// The points are read through `edited`, so a point the version has deleted is
/// never found. `views` holds one entry per image of `edited`; only their
/// cameras are read.
pub fn nearby_points(
    edited: &EditedReconstruction,
    views: &[ProjectedImage<'_>],
    image: u32,
    pixel: [f64; 2],
    options: &PointsOptions,
) -> Result<Vec<NearbyCandidate>, NearbySourceError> {
    check_query(
        edited.image_count(),
        &[("views", views.len())],
        views,
        image,
        pixel,
    )?;
    let cameras: Vec<ViewCamera<'_>> = views.iter().map(ViewCamera::new).collect();
    let index = ObservationIndex::new(edited);
    let mut out = Vec::new();
    for o in index.near(image, pixel, options.radius_px, &cameras[image as usize]) {
        if out.len() >= options.max_points {
            break;
        }
        let Some(view) = edited.point(o.point) else {
            continue;
        };
        if o.at_infinity || view.observations().len() < options.min_views {
            continue;
        }
        let Some(position) = index.position(o.point) else {
            continue;
        };
        let x = position.coords;
        let mut sightings = index.point_observations(o.point);
        let mut errors: Vec<f64> = sightings
            .iter()
            .map(|&(i, p)| {
                cameras[i as usize]
                    .project_homogeneous(&x, 1.0)
                    .map_or(f64::INFINITY, |q| {
                        let (dx, dy) = (q[0] - p[0], q[1] - p[1]);
                        (dx * dx + dy * dy).sqrt()
                    })
            })
            .collect();
        if errors.iter().any(|&e| e > options.max_reproj_px) {
            continue;
        }
        // The observation the index found in the queried image goes first.
        if let Some(q) = sightings.iter().position(|s| s.0 == image) {
            let s = sightings.remove(q);
            sightings.insert(0, s);
            let e = errors.remove(q);
            errors.insert(0, e);
        }
        let mut c = candidate(
            &cameras,
            NearbySource::Points,
            Some(o.point),
            x,
            sightings,
            errors,
            pixel,
        );
        c.point = Some(o.point);
        out.push(c);
    }
    Ok(out)
}
