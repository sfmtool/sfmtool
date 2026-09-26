// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The keypoint of each camera that lies furthest from its principal point.
//!
//! `specs/core/reconstruction/outermost-keypoint.md` is the design. A spline
//! model's domain end is a choice of how far out the lens curve is described,
//! and the one number that says how far out the photographs actually reach is
//! the outermost keypoint: the observations the reconstruction holds, and,
//! where the images' `.sift` files can be read, every feature detected in
//! them. The switch reports both, and the bundle adjustment's domain control
//! shows them beside the value.

use std::collections::HashMap;

use super::data::SfmrReconstruction;
use crate::camera::report::off_axis_angle_deg;
use crate::camera::CameraIntrinsics;

/// One keypoint, where it is and how far out.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KeypointReach {
    /// Its distance from the camera's principal point, in pixels.
    pub radius_px: f64,
    /// Its incidence angle in degrees under the camera's model: the angle
    /// between the optical axis and the ray the model gives the pixel.
    pub theta_deg: f64,
    /// The image it is in, by table index.
    pub image: usize,
    /// The pixel.
    pub xy: [f64; 2],
}

impl KeypointReach {
    /// The keypoint at `xy` in `image` under `camera`.
    fn at(camera: &CameraIntrinsics, image: usize, xy: [f64; 2]) -> Self {
        let (cx, cy) = camera.principal_point();
        Self {
            radius_px: (xy[0] - cx).hypot(xy[1] - cy),
            theta_deg: off_axis_angle_deg(camera, xy[0], xy[1]),
            image,
            xy,
        }
    }
}

/// The outermost keypoints of one camera's images.
#[derive(Debug, Clone, PartialEq)]
pub struct OutermostKeypoints {
    /// The camera's index in the reconstruction's camera table.
    pub camera: usize,
    /// The images that use it, posed or not.
    pub images: usize,
    /// The outermost of the reconstruction's own observations in those
    /// images, or `None` when they have none (or, for a `sift_files` value,
    /// when their `.sift` files could not be read).
    pub observed: Option<KeypointReach>,
    /// The outermost of every feature detected in those images, read from
    /// their `.sift` files, or `None` when none was asked for or none could be
    /// read.
    pub detected: Option<KeypointReach>,
    /// How many of those images' `.sift` files were read for `detected`.
    pub detected_images: usize,
}

/// The outermost keypoint of each of `cameras`, observed and detected, in
/// ascending camera order.
///
/// A keypoint's reach is its pixel distance from the camera's principal point,
/// and its angle is the incidence angle the camera's own model gives that pixel,
/// so a caller holding a camera after a refit or a solve measures under that
/// camera. **Observed** is over the reconstruction's observations of images that
/// use the camera: the inline keypoint where the value carries one, otherwise the
/// feature its `.sift` file holds, which needs `read_sift_files`. **Detected** is
/// over every feature of those images' `.sift` files, read when
/// `read_sift_files` is set; an image whose file cannot be read is skipped and
/// not counted in `detected_images`, so a value that sits beside no `.sift`
/// files reports `detected: None` rather than failing. A camera index past the
/// table is ignored.
///
/// # Example
///
/// ```no_run
/// use sfmtool_core::reconstruction::outermost_keypoint::outermost_keypoints;
/// # fn run(recon: &sfmtool_core::SfmrReconstruction) {
/// for camera in outermost_keypoints(recon, &[0, 1], true) {
///     if let Some(detected) = camera.detected {
///         println!(
///             "camera {}: {:.1} px, {:.1}° detected",
///             camera.camera, detected.radius_px, detected.theta_deg
///         );
///     }
/// }
/// # }
/// ```
pub fn outermost_keypoints(
    recon: &SfmrReconstruction,
    cameras: &[usize],
    read_sift_files: bool,
) -> Vec<OutermostKeypoints> {
    let table = &recon.image_table;
    let mut chosen: Vec<usize> = cameras
        .iter()
        .copied()
        .filter(|&c| c < table.cameras.len())
        .collect();
    chosen.sort_unstable();
    chosen.dedup();
    let mut out: Vec<OutermostKeypoints> = chosen
        .iter()
        .map(|&camera| OutermostKeypoints {
            camera,
            images: 0,
            observed: None,
            detected: None,
            detected_images: 0,
        })
        .collect();
    let slot_of: HashMap<usize, usize> = chosen.iter().enumerate().map(|(s, &c)| (c, s)).collect();

    // Each image's observation rows, for the images of the chosen cameras.
    let tracks = &recon.point_set.tracks;
    let mut rows_of: Vec<Vec<usize>> = vec![Vec::new(); table.images.len()];
    for (row, observation) in tracks.iter().enumerate() {
        let image = observation.image_index as usize;
        if slot_of.contains_key(&(table.images[image].camera_index as usize)) {
            rows_of[image].push(row);
        }
    }
    let inline = recon.keypoints_xy();
    let features = recon.feature_indexes();

    for (i, image) in table.images.iter().enumerate() {
        let c = image.camera_index as usize;
        let Some(&slot) = slot_of.get(&c) else {
            continue;
        };
        let camera = &table.cameras[c];
        let entry = &mut out[slot];
        entry.images += 1;
        let positions = if read_sift_files {
            sfmtool_sift_format::read_sift_positions(&recon.sift_path_for_image(i), usize::MAX).ok()
        } else {
            None
        };
        if let Some(positions) = &positions {
            entry.detected_images += 1;
            for p in positions {
                keep_outer(
                    &mut entry.detected,
                    camera,
                    i,
                    [f64::from(p[0]), f64::from(p[1])],
                );
            }
        }
        for &row in &rows_of[i] {
            let xy = match (&inline, features, &positions) {
                (Some(xy), _, _) => Some([f64::from(xy[[row, 0]]), f64::from(xy[[row, 1]])]),
                (None, Some(features), Some(positions)) => positions
                    .get(features[row] as usize)
                    .map(|p| [f64::from(p[0]), f64::from(p[1])]),
                _ => None,
            };
            if let Some(xy) = xy {
                keep_outer(&mut entry.observed, camera, i, xy);
            }
        }
    }
    out
}

/// Replace `best` with the keypoint at `xy` when it lies further out.
fn keep_outer(
    best: &mut Option<KeypointReach>,
    camera: &CameraIntrinsics,
    image: usize,
    xy: [f64; 2],
) {
    if !(xy[0].is_finite() && xy[1].is_finite()) {
        return;
    }
    let (cx, cy) = camera.principal_point();
    let radius = (xy[0] - cx).hypot(xy[1] - cy);
    if best.is_none_or(|b| radius > b.radius_px) {
        *best = Some(KeypointReach::at(camera, image, xy));
    }
}

#[cfg(test)]
mod tests;
