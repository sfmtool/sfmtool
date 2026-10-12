// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The observations a proposal compares its two cameras over: the keypoints of
//! the camera's images, read once when the proposal opens.
//!
//! The Camera Intrinsics panel's rug draws one tick per observation at its
//! incidence angle under the current camera, and the Image Detail
//! observation layer draws, for each observation of the image shown, the
//! keypoint and its reprojection under the current and the proposed camera.
//! Both read the keypoints from here, so they describe one set.

use sfmtool_core::camera::report::off_axis_angle_deg;
use sfmtool_core::SfmrReconstruction;

use super::CameraModelProposal;

/// One observation of the proposal's camera whose keypoint could be read.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct ObservedKeypoint {
    /// The image, by table index.
    pub(crate) image: usize,
    /// The point it observes, by index in the value the proposal fits.
    pub(crate) point: usize,
    /// The keypoint, in pixels.
    pub(crate) xy: [f64; 2],
}

/// One observation as the Image Detail observation layer draws it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct ObservationReprojection {
    /// The keypoint, in pixels.
    pub(crate) keypoint: [f64; 2],
    /// Where the current camera puts the observed point, or `None` where it
    /// cannot project it.
    pub(crate) current: Option<[f64; 2]>,
    /// Where the proposed camera puts it, or `None` where it cannot project
    /// it or the fit was refused.
    pub(crate) proposed: Option<[f64; 2]>,
}

/// The observations of `camera`'s images in `value` whose keypoint can be read,
/// in ascending image order.
///
/// The keypoint is the value's inline one where it carries the column, and
/// otherwise the feature the image's `.sift` file holds, each file read once.
/// An observation whose file cannot be read, or whose keypoint is not finite,
/// is left out.
pub(crate) fn observed_keypoints(
    value: &SfmrReconstruction,
    camera: usize,
) -> Vec<ObservedKeypoint> {
    let table = &value.image_table;
    let mut rows_of: Vec<Vec<usize>> = vec![Vec::new(); table.images.len()];
    for (row, observation) in value.point_set.tracks.iter().enumerate() {
        let image = observation.image_index as usize;
        if table
            .images
            .get(image)
            .is_some_and(|row| row.camera_index as usize == camera)
        {
            rows_of[image].push(row);
        }
    }
    let inline = value.keypoints_xy();
    let features = value.feature_indexes();
    let tracks = &value.point_set.tracks;
    let mut out = Vec::new();
    for (image, rows) in rows_of.iter().enumerate() {
        if rows.is_empty() {
            continue;
        }
        let positions = match (inline, features) {
            (None, Some(_)) => sfmtool_sift_format::read_sift_positions(
                &value.sift_path_for_image(image),
                usize::MAX,
            )
            .ok(),
            _ => None,
        };
        for &row in rows {
            let xy = match (inline, features, &positions) {
                (Some(xy), _, _) => Some([f64::from(xy[[row, 0]]), f64::from(xy[[row, 1]])]),
                (None, Some(features), Some(positions)) => positions
                    .get(features[row] as usize)
                    .map(|p| [f64::from(p[0]), f64::from(p[1])]),
                _ => None,
            };
            if let Some(xy) = xy.filter(|xy| xy[0].is_finite() && xy[1].is_finite()) {
                out.push(ObservedKeypoint {
                    image,
                    point: tracks[row].point_index as usize,
                    xy,
                });
            }
        }
    }
    out
}

impl CameraModelProposal {
    /// The rug: the incidence angle of each observation under the current
    /// camera, ascending, one per observation.
    pub(crate) fn rug(&self) -> &[f64] {
        &self.rug
    }

    /// Each observation of `image`, with the keypoint and the observed
    /// point's reprojection under the current and the proposed camera, through
    /// the image's stored pose. A point at infinity is reprojected as the
    /// direction it is. Empty for an image the proposal's camera did not take.
    pub(crate) fn image_reprojections(&self, image: usize) -> Vec<ObservationReprojection> {
        let Some(row) = self.value.image_table.images.get(image) else {
            return Vec::new();
        };
        if row.camera_index as usize != self.camera {
            return Vec::new();
        }
        let pose = crate::scene::cam_from_world(row);
        let proposed = self.fit.as_ref().ok().map(|refit| &refit.camera);
        let start = self.observations.partition_point(|o| o.image < image);
        self.observations[start..]
            .iter()
            .take_while(|o| o.image == image)
            .filter_map(|o| {
                let point = self.value.point_set.points.get(o.point)?;
                let project = |camera: &sfmtool_core::CameraIntrinsics| {
                    camera.project_homogeneous(&pose, point.position.coords, point.w)
                };
                Some(ObservationReprojection {
                    keypoint: o.xy,
                    current: project(&self.source),
                    proposed: proposed.and_then(project),
                })
            })
            .collect()
    }
}

/// The rug of `observations` under `camera`: their incidence angles,
/// ascending.
pub(crate) fn rug_angles(
    camera: &sfmtool_core::CameraIntrinsics,
    observations: &[ObservedKeypoint],
) -> Vec<f64> {
    let mut angles: Vec<f64> = observations
        .iter()
        .map(|o| off_axis_angle_deg(camera, o.xy[0], o.xy[1]))
        .filter(|theta| theta.is_finite())
        .collect();
    angles.sort_by(f64::total_cmp);
    angles
}
