// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Zoom-to-fit over finite points and points at infinity.
//!
//! A point at infinity stores a unit direction rather than a position, so it
//! never adds a position to the framing. When a node has finite points they
//! are framed as [`ViewportCamera::compute_zoom_to_fit`] frames any points, and
//! the bearings are ignored. When it has only bearings (a panorama), the camera
//! goes to the default starting position and looks along [`framing_bearing`].
//! See `specs/gui/viewport-navigation.md` § "Points at Infinity".

use nalgebra::{Point3, UnitQuaternion, Vector3};
use sfmtool_core::Camera;

use super::camera::ViewportCamera;
use crate::scene::FitPoints;

#[cfg(test)]
mod tests;

/// The mean resultant length at or above which the mean of the bearings is
/// the framing bearing. Directions spread evenly over a hemisphere have exactly
/// this length, so the mean is used when the bearings sit within about half the
/// sphere, and the largest cluster is used when they are spread wider.
pub(crate) const MIN_MEAN_RESULTANT_LENGTH: f64 = 0.5;

/// The angular radius of the neighbourhood the cluster count is taken over. A
/// 60° cone is a little wider than the default 45° field of view.
pub(crate) const CLUSTER_RADIUS_DEG: f64 = 30.0;

/// At most this many bearings are tried as cluster centres, taken at an even
/// stride, so the count is bounded by this many passes over the bearings.
const MAX_CLUSTER_CANDIDATES: usize = 512;

/// Where a zoom-to-fit leaves the camera.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct FitEnd {
    pub(crate) position: Point3<f64>,
    pub(crate) orientation: UnitQuaternion<f64>,
    pub(crate) target_distance: f64,
    pub(crate) world_up: Vector3<f64>,
}

/// The direction to look along to frame `bearings`, or `None` when there are
/// none with a direction.
///
/// The normalized mean when the mean resultant length is at least
/// [`MIN_MEAN_RESULTANT_LENGTH`]; otherwise the normalized mean of the bearings
/// within [`CLUSTER_RADIUS_DEG`] of the candidate whose neighbourhood holds the
/// most of them, the first such candidate winning a tie.
pub(crate) fn framing_bearing(bearings: &[Vector3<f64>]) -> Option<Vector3<f64>> {
    let unit: Vec<Vector3<f64>> = bearings
        .iter()
        .filter_map(|b| {
            let norm = b.norm();
            (norm.is_finite() && norm > 1e-12).then(|| b / norm)
        })
        .collect();
    if unit.is_empty() {
        return None;
    }
    let sum = unit.iter().fold(Vector3::zeros(), |acc, b| acc + b);
    if sum.norm() / unit.len() as f64 >= MIN_MEAN_RESULTANT_LENGTH {
        return Some(sum.normalize());
    }
    let cos_radius = CLUSTER_RADIUS_DEG.to_radians().cos();
    let stride = unit.len().div_ceil(MAX_CLUSTER_CANDIDATES);
    let mut best: Option<(usize, Vector3<f64>)> = None;
    for candidate in unit.iter().step_by(stride) {
        let (count, near) = unit
            .iter()
            .filter(|b| b.dot(candidate) >= cos_radius)
            .fold((0usize, Vector3::zeros()), |(n, acc), b| (n + 1, acc + b));
        if best.is_none_or(|(most, _)| count > most) {
            best = Some((count, near));
        }
    }
    // The candidate is in its own neighbourhood, and bearings within 30° of
    // one direction cannot cancel, so the sum has a direction.
    best.map(|(_, near)| near.normalize())
}

impl ViewportCamera {
    /// Where zoom-to-fit over `points` leaves the camera, or `None` when there
    /// is nothing to frame.
    ///
    /// Finite positions are framed keeping the current orientation and
    /// `world_up`. With none, the camera goes to the default starting position
    /// and looks along [`framing_bearing`] of the bearings; `keep_level` (the
    /// viewport's Maintain Z-up) makes `world_up` +Z for that turn, and
    /// otherwise the current `world_up` is kept.
    pub(crate) fn compute_fit(
        &self,
        points: &FitPoints,
        aspect: f64,
        keep_level: bool,
    ) -> Option<FitEnd> {
        if !points.positions.is_empty() {
            let (position, target_distance) =
                self.compute_zoom_to_fit(&points.positions, aspect)?;
            return Some(FitEnd {
                position,
                orientation: self.camera.orientation,
                target_distance,
                world_up: self.world_up,
            });
        }
        let forward = framing_bearing(&points.bearings)?;
        let start = ViewportCamera::default();
        let world_up = if keep_level {
            Vector3::z()
        } else {
            self.world_up
        };
        Some(FitEnd {
            position: start.camera.position,
            orientation: Camera::orientation_from_forward(forward, world_up),
            target_distance: start.camera.target_distance,
            world_up,
        })
    }

    /// Put the camera at `end` at once.
    pub(crate) fn apply_fit(&mut self, end: &FitEnd) {
        self.camera.position = end.position;
        self.camera.orientation = end.orientation;
        self.camera.target_distance = end.target_distance;
        self.world_up = end.world_up;
    }
}
