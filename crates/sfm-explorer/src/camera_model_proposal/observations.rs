// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The observations a proposal compares its two cameras over: the keypoints of
//! the camera's images, read once when the proposal opens.
//!
//! The Camera Intrinsics panel's rug draws one tick per observation at its
//! incidence angle under the current camera, and the Image Detail
//! observation layer draws, for each observation of the image shown, the
//! keypoint and its reprojection under the current and the proposed camera.
//! Both read the keypoints from here, so they describe one set. The same pass
//! over the images' `.sift` files gives the outermost keypoint extent the
//! proposal strip shows beside the spline domain, so each file is read once.

use std::path::Path;

use sfmtool_core::camera::report::off_axis_angle_deg;
use sfmtool_core::reconstruction::outermost_keypoint::KeypointReach;
use sfmtool_core::SfmrReconstruction;

use super::{CameraModelProposal, KeypointExtent};

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

/// The outermost keypoint extent of `camera` over `base`, and the observations
/// of `camera`'s images in `value` whose keypoint can be read, in ascending
/// image order, from **one** read of each image's `.sift` file.
///
/// `base` is the version's base, which the extent is measured over as the
/// switch reports it (`outermost_keypoints` with its `.sift` files read), and
/// `value` the value the fits read: the base itself, or its materialisation.
/// The extent is the one `outermost_keypoints` gives: the outermost observation
/// and the outermost detected feature, by radius from the principal point. An
/// observation's keypoint is the value's inline one where it carries the
/// column, and otherwise the feature the image's `.sift` file holds. An
/// observation whose file cannot be read, or whose keypoint is not finite, is
/// left out.
pub(crate) fn read_camera_keypoints(
    base: &SfmrReconstruction,
    value: &SfmrReconstruction,
    camera: usize,
) -> (KeypointExtent, Vec<ObservedKeypoint>) {
    read_camera_keypoints_with(base, value, camera, |path| {
        sfmtool_sift_format::read_sift_positions(path, usize::MAX).ok()
    })
}

/// [`read_camera_keypoints`] with the `.sift` read given, so a test can count
/// the reads.
pub(super) fn read_camera_keypoints_with(
    base: &SfmrReconstruction,
    value: &SfmrReconstruction,
    camera: usize,
    mut read: impl FnMut(&Path) -> Option<Vec<[f32; 2]>>,
) -> (KeypointExtent, Vec<ObservedKeypoint>) {
    let mut extent = KeypointExtent::default();
    let mut out = Vec::new();
    let table = &base.image_set;
    let Some(lens) = table.cameras.get(camera) else {
        return (extent, out);
    };
    let base_rows = rows_by_image(base, camera);
    let value_rows = rows_by_image(value, camera);
    for (image, row) in table.images.iter().enumerate() {
        if row.camera_index as usize != camera {
            continue;
        }
        let positions = read(&base.sift_path_for_image(image));
        let feature = |positions: &Option<Vec<[f32; 2]>>, index: u32| {
            positions
                .as_ref()
                .and_then(|p| p.get(index as usize))
                .map(|p| [f64::from(p[0]), f64::from(p[1])])
        };
        if let Some(positions) = &positions {
            for p in positions {
                keep_outer(
                    &mut extent.detected,
                    lens,
                    image,
                    [f64::from(p[0]), f64::from(p[1])],
                );
            }
        }
        let keypoint = |recon: &SfmrReconstruction, row: usize| match (
            recon.keypoints_xy(),
            recon.feature_indexes(),
        ) {
            (Some(xy), _) => Some([f64::from(xy[[row, 0]]), f64::from(xy[[row, 1]])]),
            (None, Some(features)) => feature(&positions, features[row]),
            _ => None,
        };
        for &r in base_rows.get(image).into_iter().flatten() {
            if let Some(xy) = keypoint(base, r) {
                keep_outer(&mut extent.observed, lens, image, xy);
            }
        }
        for &r in value_rows.get(image).into_iter().flatten() {
            if let Some(xy) = keypoint(value, r).filter(|xy| xy[0].is_finite() && xy[1].is_finite())
            {
                out.push(ObservedKeypoint {
                    image,
                    point: value.point_set.tracks[r].point_index as usize,
                    xy,
                });
            }
        }
    }
    (extent, out)
}

/// The observation rows of each image of `camera` in `recon`, by image.
fn rows_by_image(recon: &SfmrReconstruction, camera: usize) -> Vec<Vec<usize>> {
    let table = &recon.image_set;
    let mut rows_of: Vec<Vec<usize>> = vec![Vec::new(); table.images.len()];
    for (row, observation) in recon.point_set.tracks.iter().enumerate() {
        let image = observation.image_index as usize;
        if table
            .images
            .get(image)
            .is_some_and(|row| row.camera_index as usize == camera)
        {
            rows_of[image].push(row);
        }
    }
    rows_of
}

/// Replace `best` with the keypoint at `xy` when it lies further out, by the
/// rule `outermost_keypoints` keeps: finite keypoints only, and a strictly
/// larger radius from the principal point.
fn keep_outer(
    best: &mut Option<KeypointReach>,
    camera: &sfmtool_core::CameraIntrinsics,
    image: usize,
    xy: [f64; 2],
) {
    if !(xy[0].is_finite() && xy[1].is_finite()) {
        return;
    }
    let (cx, cy) = camera.principal_point();
    let radius = (xy[0] - cx).hypot(xy[1] - cy);
    if best.is_none_or(|b| radius > b.radius_px) {
        *best = Some(KeypointReach {
            radius_px: radius,
            theta_deg: off_axis_angle_deg(camera, xy[0], xy[1]),
            image,
            xy,
        });
    }
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
        let Some(row) = self.value.image_set.images.get(image) else {
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
