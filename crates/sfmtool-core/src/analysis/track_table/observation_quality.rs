// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The track table columns read per observation of a track: the largest
//! reprojection error, the viewing angles of the patch, the blur-matched score
//! against the stored bitmap, and the self-similarity of the point's patch as
//! its reference view sees it.
//!
//! See `specs/core/analysis/track-table.md` § "The columns".

use crate::geometry::RigidTransform;
use crate::patch::normal_refine::viewing_angle;
use crate::patch::self_similarity::{zncc_self_similarity_radius, PatchTile, SelfSimilarityParams};
use crate::patch::stored_bitmap::stored_bitmap_planes;
use crate::reconstruction::data::observation_reprojection_error;
use crate::reconstruction::{ImageSet, PointView};
use crate::RotQuaternion;

/// The largest reprojection error among the track's observations, in pixels:
/// each observation's inline keypoint against the point's projection into its
/// image, a point at infinity projected through its bearing. An observation
/// whose point does not project (behind the camera, or outside the lens's
/// domain) is skipped, as the stored RMS error skips it; NaN when none
/// projects or the reconstruction holds no inline keypoints.
pub(super) fn max_error_px(image_set: &ImageSet, view: &PointView<'_>) -> f64 {
    let point = view.point();
    let at_infinity = point.is_at_infinity();
    let mut largest = f64::NAN;
    for (k, obs) in view.observations().iter().enumerate() {
        let Some(xy) = view.keypoint_xy(k) else {
            continue;
        };
        let Some(image) = image_set.images.get(obs.image_index as usize) else {
            continue;
        };
        let camera = &image_set.cameras[image.camera_index as usize];
        if let Some(error) = observation_reprojection_error(
            &image.quaternion_wxyz,
            &image.translation_xyz,
            camera,
            &point.position,
            at_infinity,
            [f64::from(xy[0]), f64::from(xy[1])],
        ) {
            largest = largest.max(error);
        }
    }
    largest
}

/// The largest viewing angle of the point's patch over its observations, in
/// degrees, and how many observations see the patch at `90°` or more
/// (back-facing); `None` for a point with no patch frame.
///
/// Each angle is the one Track View shows for the row
/// ([`viewing_angle`]): read at the patch re-anchored on the observation's
/// keypoint where the observation has one and its ray meets the patch's
/// plane, and at the patch's stored centre otherwise, as a render without a
/// keypoint is. An observation whose camera sits on the patch's centre has no
/// angle and is skipped. The largest is NaN when no observation has an angle.
pub(super) fn viewing_angles(image_set: &ImageSet, view: &PointView<'_>) -> Option<(f64, f64)> {
    let patch = view.patch_frame()?;
    let mut largest = f64::NAN;
    let mut back_facing = 0usize;
    for (k, obs) in view.observations().iter().enumerate() {
        let Some(image) = image_set.images.get(obs.image_index as usize) else {
            continue;
        };
        let camera = &image_set.cameras[image.camera_index as usize];
        let cam_from_world = RigidTransform::new(
            RotQuaternion::from_nalgebra(image.quaternion_wxyz),
            image.translation_xyz,
        );
        let anchored = view.keypoint_xy(k).and_then(|xy| {
            patch.anchored_at_keypoint(
                camera,
                &cam_from_world,
                [f64::from(xy[0]), f64::from(xy[1])],
            )
        });
        let Some(angle) = viewing_angle(anchored.as_ref().unwrap_or(&patch), &cam_from_world)
        else {
            continue;
        };
        largest = largest.max(angle.angle_deg);
        // The reference-view rule's test for a back-facing view.
        if angle.angle_deg >= 90.0 {
            back_facing += 1;
        }
    }
    Some((largest, back_facing as f64))
}

/// The lowest blur-matched ZNCC against the stored bitmap among the track's
/// observation readings, skipping the rows whose score was not read (NaN);
/// NaN when none was read or the reconstruction holds no readings.
pub(super) fn min_blur_matched_bitmap_zncc(view: &PointView<'_>) -> f64 {
    view.observation_readings()
        .and_then(|rows| {
            rows.iter()
                .map(|r| f64::from(r.blur_matched_bitmap_zncc))
                .filter(|z| !z.is_nan())
                .reduce(f64::min)
        })
        .unwrap_or(f64::NAN)
}

/// The `[semi-major, semi-minor]` axes of a self-similarity ellipse, in
/// patch-grid px, and per axis whether the true length may be larger.
#[derive(Copy, Clone, Debug, PartialEq)]
pub(super) struct Axes {
    pub(super) lengths: [f64; 2],
    pub(super) is_at_least: [bool; 2],
}

/// The axes of the ZNCC self-similarity ellipse of the point's patch as its
/// reference view sees it, with their lower-bound flags; NaN and `false` in
/// both where there is no reading.
///
/// The stored reading at the point's reference observation, where the
/// reconstruction holds readings, the point names a reference observation
/// (not `-1`) and that row is measured: the reading of the reference
/// observation's render, which is the render the stored bitmap holds.
/// Otherwise the whole-bitmap reading of the point's stored patch bitmap, the
/// overlap reading with the default settings the stored readings and the
/// `embed-patches` cull use ([`SelfSimilarityParams::default`]), a sample
/// whose alpha is `0` carrying no data. A bitmap with no sample carrying data
/// (a zero row) or under 3 px a side has no reading.
pub(super) fn reference_self_similarity_axes(view: &PointView<'_>) -> Axes {
    let stored = view
        .observation_readings()
        .zip(view.reference_observation())
        .and_then(|(rows, reference)| rows.get(usize::try_from(reference).ok()?))
        .filter(|row| row.is_measured());
    if let Some(row) = stored {
        return Axes {
            lengths: row.ellipse_axes.map(f64::from),
            is_at_least: row.ellipse_axes_is_at_least,
        };
    }
    view.patch_bitmap()
        .and_then(stored_bitmap_planes)
        .filter(|planes| planes.side >= 3 && planes.data.iter().any(|&d| d))
        .map_or(
            Axes {
                lengths: [f64::NAN; 2],
                is_at_least: [false; 2],
            },
            |planes| {
                let side = planes.side;
                let ellipse = zncc_self_similarity_radius(
                    &PatchTile {
                        values: &planes.values,
                        channels: planes.channels.min(3),
                        width: side,
                        height: side,
                    },
                    Some(&planes.data),
                    [0, 0, side, side],
                    &SelfSimilarityParams::default(),
                )
                .ellipse;
                Axes {
                    lengths: ellipse.axes,
                    is_at_least: ellipse.axes_is_at_least,
                }
            },
        )
}
