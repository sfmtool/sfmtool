// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The image side of a reconstruction: the cameras, the posed images, and
//! everything measured per image.

use std::sync::Arc;

use ndarray::Array4;
use sfmr_format::{DepthStatistics, RigFrameData};

use crate::camera::CameraIntrinsics;

use super::SfmrImage;

/// The cameras, the posed images, and the per-image columns -- one half of a
/// [`SfmrReconstruction`](super::SfmrReconstruction), the half a point never
/// belongs to.
///
/// Everything here is addressed by image index, so an operation that adds,
/// drops or reorders images works on this table and an operation that only
/// moves points leaves it alone. The columns are parallel to `images`, except
/// `cameras`, which `SfmrImage::camera_index` indexes into, and
/// `rig_frame_data`, which groups images into rig frames.
#[derive(Clone)]
pub struct ImageTable {
    /// Camera intrinsic parameters, indexed by `SfmrImage::camera_index`.
    pub cameras: Vec<CameraIntrinsics>,
    /// Registered images with poses.
    pub images: Vec<SfmrImage>,
    /// `(N, THUMBNAIL_SIZE, THUMBNAIL_SIZE, 3)` RGB thumbnails of the source
    /// images (see [`crate::THUMBNAIL_SIZE`]).
    ///
    /// Behind an [`Arc`] because it is one of the two columns that dominate a
    /// reconstruction's memory, so two reconstructions that agree on their
    /// thumbnails share one copy. Nothing writes through it: a producer that
    /// changes the thumbnails builds a new array and wraps it.
    pub thumbnails_y_x_rgb: Arc<Array4<u8>>,
    /// Per-image depth statistics.
    pub depth_statistics: DepthStatistics,
    /// Depth histogram counts: `depth_histogram_counts[i]` has
    /// `num_histogram_buckets` entries.
    pub depth_histogram_counts: Vec<Vec<u32>>,
    /// Rig definitions and frame groupings. `None` when no multi-camera rigs.
    pub rig_frame_data: Option<RigFrameData>,
}

impl ImageTable {
    /// Number of registered images.
    pub fn image_count(&self) -> usize {
        self.images.len()
    }

    /// Number of camera models.
    pub fn camera_count(&self) -> usize {
        self.cameras.len()
    }

    /// The camera an image was taken with.
    pub fn camera_for_image(&self, image_index: usize) -> &CameraIntrinsics {
        &self.cameras[self.images[image_index].camera_index as usize]
    }

    /// The unit world-space ray `pixel` states in image `image_index`, or
    /// `None` when the index is past the table or the camera model has no ray
    /// there.
    ///
    /// The pose and the lens are both this table's, so every edit that
    /// triangulates a track -- a shortened one, or one whose camera moved --
    /// assembles its rays here rather than each spelling the pixel-to-world
    /// chain out again.
    pub fn world_ray(&self, image_index: usize, pixel: [f64; 2]) -> Option<nalgebra::Vector3<f64>> {
        let image = self.images.get(image_index)?;
        let camera = self.cameras.get(image.camera_index as usize)?;
        let ray = camera.pixel_to_ray(pixel[0], pixel[1]);
        let cam = nalgebra::Vector3::new(ray[0], ray[1], ray[2]);
        if !cam.iter().all(|c| c.is_finite()) || cam.norm() <= 0.0 {
            return None;
        }
        Some((image.quaternion_wxyz.inverse() * cam).normalize())
    }

    /// The distance from the camera-cloud centroid to `position`: what a patch
    /// frame's angular extent is multiplied by to become a world one there, and
    /// what a world extent is divided by to become an angular one.
    ///
    /// The reference is the mean of **every** camera centre in the table rather
    /// than of the ones that see any particular point, because that is the
    /// reference
    /// [`SfmrReconstruction::materialize_points_at_infinity`](super::SfmrReconstruction::materialize_points_at_infinity)
    /// places a bearing at. Every edit that resizes a patch frame measures from
    /// here, so their conversions cancel exactly rather than to within which
    /// images happen to be in a track.
    pub fn placement_scale(&self, position: &nalgebra::Point3<f64>) -> f64 {
        let mut centroid = nalgebra::Vector3::zeros();
        for image in &self.images {
            centroid += image.camera_center().coords;
        }
        if !self.images.is_empty() {
            centroid /= self.images.len() as f64;
        }
        (position.coords - centroid).norm()
    }
}
