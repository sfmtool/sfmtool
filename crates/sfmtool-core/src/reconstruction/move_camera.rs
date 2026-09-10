// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Replace one image's pose, and settle the tracks it observes.
//!
//! `specs/core/reconstruction/move-camera.md` is the design. The function here
//! is pure: an [`SfmrReconstruction`] goes in, a new one and a report come out,
//! and the input is left exactly as it was.

use nalgebra::{Point3, Vector3};

use super::bundle_adjust::{is_posed, rescale_patch_frame};
use super::data::{observation_reprojection_error, SfmrReconstruction};
use super::triangulation::triangulate_track;
use crate::camera::CameraIntrinsics;
use crate::geometry::resect_images::scene_scale;
use crate::numeric::median_in_place;
use crate::{RotQuaternion, Se3Transform};

/// Why a camera could not be moved. Every variant names what did not hold,
/// because the caller is a menu entry that has to say so in one sentence.
#[derive(Debug, Clone, PartialEq)]
pub enum MoveCameraError {
    /// The image index is past the reconstruction's image table.
    ImageOutOfRange {
        /// The index named.
        image: usize,
        /// How many images the reconstruction holds.
        image_count: usize,
    },
    /// The image carries no pose, so there is none to replace: a `.sfmr` row
    /// whose rotation or translation is a non-finite placeholder.
    NoPose(usize),
    /// The pose asked for is not a finite rigid placement.
    InvalidPose,
}

impl std::fmt::Display for MoveCameraError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MoveCameraError::ImageOutOfRange { image, image_count } => write!(
                f,
                "image {image} is past the {image_count} images of the reconstruction"
            ),
            MoveCameraError::NoPose(image) => {
                write!(f, "image {image} carries no pose to move")
            }
            MoveCameraError::InvalidPose => write!(
                f,
                "the pose is not a finite placement: its rotation must be a unit quaternion \
                 and its translation finite"
            ),
        }
    }
}

impl std::error::Error for MoveCameraError {}

/// What one moved camera did.
#[derive(Debug, Clone, PartialEq)]
pub struct MoveCameraReport {
    /// The image whose pose was replaced.
    pub image: usize,
    /// Angle between the stored and the new world-to-camera rotation, degrees.
    pub rotation_deg: f64,
    /// Distance between the stored and the new camera centre, in the
    /// reconstruction's own units.
    pub translation: f64,
    /// [`MoveCameraReport::translation`] in units of the capture's own length
    /// scale -- the median over images of that image's median
    /// camera-to-structure distance -- and `None` where there is no finite
    /// structure to measure against.
    pub translation_scene: Option<f64>,
    /// Tracks that observe this image. Equals the three counts below summed.
    pub observed: usize,
    /// Of those, the ones re-triangulated at the new pose.
    pub retriangulated: usize,
    /// Of those, the ones left exactly where they stood: a track whose
    /// triangulation failed, a bearing more than one image sees, a finite point
    /// this image alone sees, and any track whose observations carry no pixel.
    pub kept: usize,
    /// Of those, the single-view bearings that rotated with the camera.
    pub rotated_bearings: usize,
    /// This image's median and 90th-percentile reprojection residual before the
    /// move, in pixels; `None` when no observation of it projects, which is
    /// what a value carrying no inline keypoints gives.
    pub residual_before_px: Option<[f64; 2]>,
    /// The same two numbers after it, over the same observations at the
    /// geometry that came back.
    pub residual_after_px: Option<[f64; 2]>,
}

/// One observation, as the residual readout measures it: where the point is,
/// and which pixel saw it.
#[derive(Debug, Clone, PartialEq)]
pub struct ReprojectionSample {
    /// The point's world position, or its unit world direction when it is a
    /// bearing.
    pub position: Point3<f64>,
    /// Whether `position` is a direction rather than a location.
    pub at_infinity: bool,
    /// The pixel the observation stands at.
    pub keypoint: [f64; 2],
}

/// Replace image `image`'s pose with `world_from_camera`, and settle the tracks
/// it observes.
///
/// `world_from_camera` places the camera in the reconstruction's **own** frame:
/// its rotation carries camera axes onto world axes and its translation is the
/// camera centre. A camera pose has no scale, so [`Se3Transform::scale`] is not
/// read; the type is the one a viewer's transform arithmetic produces.
///
/// The pose is the caller's decision and the points are its consequence, so
/// each track this image observes is settled on its own evidence and no single
/// track's failure refuses the move:
///
/// - a **finite point with two or more observations that all carry a pixel** is
///   re-triangulated from the value's own poses and lenses, the moved one
///   included, by [`triangulate_track`] -- the same solve
///   [`remove_observation`](super::remove_observation::remove_observation) runs
///   over a shortened track. Its patch frame is rescaled by its
///   placement-distance ratio so the patch keeps its angular size, and its
///   stored `error` becomes the RMS of its own residuals at the new geometry;
/// - a **bearing whose only observation is this image** rotates with the
///   camera: its direction becomes the moved camera's ray through its keypoint;
/// - everything else **keeps its position**: a track whose triangulation
///   failed, a bearing more than one image sees, a finite point this image
///   alone sees, and any track whose observations carry no pixel. A
///   `sift_files` value without the optional inline keypoint column states no
///   ray, so its camera moves and its points stand.
///
/// The derived indexes are rebuilt, so the value that comes back is complete.
///
/// # Example
///
/// ```no_run
/// use sfmtool_core::reconstruction::move_camera::move_camera;
/// use sfmtool_core::{RotQuaternion, Se3Transform};
/// # fn run(recon: &sfmtool_core::SfmrReconstruction)
/// # -> Result<(), Box<dyn std::error::Error>> {
/// let pose = Se3Transform::new(RotQuaternion::identity(), nalgebra::Vector3::zeros(), 1.0);
/// let (next, report) = move_camera(recon, 7, &pose)?;
/// println!(
///     "{:.2} deg, {} points re-solved",
///     report.rotation_deg, report.retriangulated
/// );
/// # Ok(())
/// # }
/// ```
pub fn move_camera(
    recon: &SfmrReconstruction,
    image: usize,
    world_from_camera: &Se3Transform,
) -> Result<(SfmrReconstruction, MoveCameraReport), MoveCameraError> {
    let image_count = recon.image_table.images.len();
    if image >= image_count {
        return Err(MoveCameraError::ImageOutOfRange { image, image_count });
    }
    let stored = &recon.image_table.images[image];
    if !is_posed(&stored.quaternion_wxyz, &stored.translation_xyz) {
        return Err(MoveCameraError::NoPose(image));
    }
    let rotation = *world_from_camera.rotation.as_nalgebra();
    let centre = world_from_camera.translation;
    let unit = (rotation.coords.norm() - 1.0).abs() <= 1e-6;
    if !unit || !centre.iter().all(|c| c.is_finite()) {
        return Err(MoveCameraError::InvalidPose);
    }

    // World-to-camera is what the row stores: the inverse rotation, and the
    // translation that carries the new centre onto the camera origin.
    let cam_from_world = rotation.inverse();
    let stored_rotation = stored.quaternion_wxyz;
    let stored_centre = stored.camera_center();
    let residual_before_px = residual_quantiles_px(
        recon.image_table.camera_for_image(image),
        &pose_of(recon, image),
        &image_reprojection_samples(recon, image),
    );

    let mut out = recon.clone();
    {
        let row = &mut out.image_table.images[image];
        row.quaternion_wxyz = cam_from_world;
        row.translation_xyz = -(cam_from_world * centre);
    }

    // The tracks this image observes. The CSR track array is sorted by point,
    // so one walk finds each of them once.
    let observed: Vec<u32> = {
        let mut seen: Vec<u32> = recon
            .point_set
            .tracks
            .iter()
            .filter(|t| t.image_index as usize == image)
            .map(|t| t.point_index)
            .collect();
        seen.dedup();
        seen
    };

    let scale = scene_scale(recon);
    let translation = (Point3::from(centre) - stored_centre).norm();
    let mut report = MoveCameraReport {
        image,
        rotation_deg: cam_from_world
            .rotation_to(&stored_rotation)
            .angle()
            .to_degrees(),
        translation,
        translation_scene: scale.map(|s| translation / s),
        observed: observed.len(),
        retriangulated: 0,
        kept: 0,
        rotated_bearings: 0,
        residual_before_px,
        residual_after_px: None,
    };

    let keypoints = recon.point_set.keypoints_xy();
    for &point in &observed {
        let p = point as usize;
        let start = recon.point_set.observation_offsets[p];
        let end = recon.point_set.observation_offsets[p + 1];
        let rows = &recon.point_set.tracks[start..end];
        // Every ray this track states at the **new** poses, or nothing when one
        // observation carries no pixel or falls outside its lens.
        let rays: Option<Vec<(Vector3<f64>, Point3<f64>)>> = keypoints.and_then(|xy| {
            rows.iter()
                .enumerate()
                .map(|(k, row)| {
                    let pixel = [f64::from(xy[[start + k, 0]]), f64::from(xy[[start + k, 1]])];
                    let at = row.image_index as usize;
                    let ray = out.image_table.world_ray(at, pixel)?;
                    Some((ray, out.image_table.images[at].camera_center()))
                })
                .collect()
        });

        // A bearing this image alone sees is the one track a single ray still
        // states: its direction becomes the moved camera's ray through its
        // keypoint. One more image sees it and the rays disagree about a
        // rotation nothing asked them for, so it stands.
        if recon.point_set.points[p].is_at_infinity() {
            match (rows.len(), rays.as_ref()) {
                (1, Some(rays)) => {
                    out.point_set.points[p].position = Point3::from(rays[0].0);
                    report.rotated_bearings += 1;
                }
                _ => report.kept += 1,
            }
            continue;
        }
        let Some(rays) = rays.filter(|rays| rays.len() >= 2) else {
            report.kept += 1;
            continue;
        };
        let dirs: Vec<Vector3<f64>> = rays.iter().map(|(d, _)| *d).collect();
        let centers: Vec<Point3<f64>> = rays.iter().map(|(_, c)| *c).collect();
        let Some(tri) = triangulate_track(&dirs, &centers) else {
            // One track that will not re-triangulate does not refuse a move the
            // reviewer has already made; it keeps the position it had.
            report.kept += 1;
            continue;
        };
        let before = recon.point_set.points[p].position;
        rescale_patch_frame(
            &mut out,
            p,
            Some(recon.image_table.placement_scale(&before)),
            Some(&tri.point),
        );
        out.point_set.points[p].position = tri.point;
        out.point_set.points[p].error = track_rms_px(&out, p) as f32;
        report.retriangulated += 1;
    }

    out.rebuild_derived_fields();
    report.residual_after_px = residual_quantiles_px(
        out.image_table.camera_for_image(image),
        &pose_of(&out, image),
        &image_reprojection_samples(&out, image),
    );
    Ok((out, report))
}

/// Every observation image `image` makes, as the residual readout measures
/// them: the point where the value has it, and the pixel that saw it.
///
/// Empty when the value carries no inline keypoint column, which is what makes
/// the readout `None` there rather than zero.
pub fn image_reprojection_samples(
    recon: &SfmrReconstruction,
    image: usize,
) -> Vec<ReprojectionSample> {
    let Some(xy) = recon.point_set.keypoints_xy() else {
        return Vec::new();
    };
    recon
        .point_set
        .tracks
        .iter()
        .enumerate()
        .filter(|(_, row)| row.image_index as usize == image)
        .map(|(row_index, row)| {
            let point = &recon.point_set.points[row.point_index as usize];
            ReprojectionSample {
                position: point.position,
                at_infinity: point.is_at_infinity(),
                keypoint: [f64::from(xy[[row_index, 0]]), f64::from(xy[[row_index, 1]])],
            }
        })
        .collect()
}

/// [`image_reprojection_samples`] over an edited value, which is what a caller
/// holding a version rather than a file has.
///
/// The overlay is what the reader is looking at, so the walk is over the live
/// points and their tracks: a point the version deleted contributes nothing and
/// one it added contributes its own sighting.
pub fn edited_image_reprojection_samples(
    edited: &super::edited::EditedReconstruction,
    image: usize,
) -> Vec<ReprojectionSample> {
    let mut samples = Vec::new();
    for index in edited.live_indexes() {
        let Some(view) = edited.point(index) else {
            continue;
        };
        for (k, observation) in view.observations().iter().enumerate() {
            if observation.image_index as usize != image {
                continue;
            }
            let Some(keypoint) = view.keypoint_xy(k) else {
                continue;
            };
            samples.push(ReprojectionSample {
                position: view.point().position,
                at_infinity: view.point().is_at_infinity(),
                keypoint: [f64::from(keypoint[0]), f64::from(keypoint[1])],
            });
        }
    }
    samples
}

/// The median and 90th-percentile reprojection residual of `samples` under
/// `world_from_camera`, in pixels, or `None` when none of them projects.
///
/// The pose is the one being *asked about* rather than one a value holds, so a
/// caller steering a camera by hand measures a pending placement against the
/// pixels already stored. The 90th percentile accompanies the median because
/// half a photograph agreeing is not the same claim as all of it agreeing, and
/// it is the tail that says which.
pub fn residual_quantiles_px(
    camera: &CameraIntrinsics,
    world_from_camera: &Se3Transform,
    samples: &[ReprojectionSample],
) -> Option<[f64; 2]> {
    let rotation = world_from_camera.rotation.as_nalgebra().inverse();
    let translation = -(rotation * world_from_camera.translation);
    let mut residuals: Vec<f64> = samples
        .iter()
        .filter_map(|sample| {
            observation_reprojection_error(
                &rotation,
                &translation,
                camera,
                &sample.position,
                sample.at_infinity,
                sample.keypoint,
            )
            .filter(|r| r.is_finite())
        })
        .collect();
    if residuals.is_empty() {
        return None;
    }
    residuals.sort_unstable_by(f64::total_cmp);
    let p90 = quantile_of_sorted(&residuals, 0.9);
    Some([median_in_place(&mut residuals), p90])
}

/// The world-from-camera pose image `image` stands at, in the value's own
/// frame: the form [`move_camera`] takes and [`residual_quantiles_px`]
/// measures against.
pub fn pose_of(recon: &SfmrReconstruction, image: usize) -> Se3Transform {
    let row = &recon.image_table.images[image];
    Se3Transform::new(
        RotQuaternion::from_nalgebra(row.quaternion_wxyz.inverse()),
        row.camera_center().coords,
        1.0,
    )
}

/// Linear-interpolated quantile of an ascending, non-empty slice.
///
/// The convention `numpy.quantile` takes, so a number printed beside a median
/// means what a reader checking it in a notebook would get.
fn quantile_of_sorted(sorted: &[f64], q: f64) -> f64 {
    let position = q * (sorted.len() - 1) as f64;
    let low = position.floor() as usize;
    let high = position.ceil() as usize;
    let t = position - low as f64;
    sorted[low] * (1.0 - t) + sorted[high] * t
}

/// The RMS of point `p`'s own reprojection residuals at `recon`'s geometry, or
/// its stored error when nothing about it projects.
fn track_rms_px(recon: &SfmrReconstruction, p: usize) -> f64 {
    let Some(xy) = recon.point_set.keypoints_xy() else {
        return f64::from(recon.point_set.points[p].error);
    };
    let point = &recon.point_set.points[p];
    let start = recon.point_set.observation_offsets[p];
    let end = recon.point_set.observation_offsets[p + 1];
    let mut sum = 0.0;
    let mut count = 0usize;
    for (k, row) in recon.point_set.tracks[start..end].iter().enumerate() {
        let at = row.image_index as usize;
        let image = &recon.image_table.images[at];
        let observed = [f64::from(xy[[start + k, 0]]), f64::from(xy[[start + k, 1]])];
        if let Some(r) = observation_reprojection_error(
            &image.quaternion_wxyz,
            &image.translation_xyz,
            recon.image_table.camera_for_image(at),
            &point.position,
            point.is_at_infinity(),
            observed,
        )
        .filter(|r| r.is_finite())
        {
            sum += r * r;
            count += 1;
        }
    }
    if count == 0 {
        return f64::from(point.error);
    }
    (sum / count as f64).sqrt()
}

#[cfg(test)]
mod tests;
