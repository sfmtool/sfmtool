// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The image side of a reconstruction: the cameras, the posed images, and
//! everything measured per image.

use std::borrow::Cow;
use std::sync::Arc;

use ndarray::Array4;
use sfmtool_sfmr_format::{
    parse_per_image_adjustment_names, DepthStatistics, RigFrameData, PER_IMAGE_ADJUSTMENT_NAMES,
};

use crate::camera::{CameraIntrinsics, ImageIntrinsicsList, IntrinsicAdjustment};
use crate::reconstruction::bundle_adjust::is_posed;

use super::SfmrImage;

/// Which parts of one camera's intrinsics vary per image: the in-memory form
/// of the camera's `per_image_adjustments` key in an `.sfmr` file.
///
/// A flag that is set says the camera's images carry their own adjustment of
/// that parameter, so a zero there was solved for rather than never enabled.
/// The default varies nothing.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PerImageAdjustments {
    /// Each image of the camera has its own focal scale.
    pub focal: bool,
    /// Each image of the camera has its own principal-point offset.
    pub principal_point: bool,
}

impl PerImageAdjustments {
    /// Whether the camera varies anything per image.
    pub fn any(&self) -> bool {
        self.focal || self.principal_point
    }

    /// The flags a camera's stored `per_image_adjustments` list names, or a
    /// message saying why the list is not one the format allows.
    pub fn from_names<S: AsRef<str>>(names: &[S]) -> Result<Self, String> {
        let [focal, principal_point] = parse_per_image_adjustment_names(names)?;
        Ok(Self {
            focal,
            principal_point,
        })
    }

    /// The camera's stored `per_image_adjustments` list: the set flags'
    /// names, in the format's order, and empty when none is set.
    pub fn names(&self) -> Vec<String> {
        [self.focal, self.principal_point]
            .into_iter()
            .zip(PER_IMAGE_ADJUSTMENT_NAMES)
            .filter(|&(set, _)| set)
            .map(|(_, name)| name.to_string())
            .collect()
    }
}

/// The cameras, the posed images, and the per-image columns -- one half of a
/// [`SfmrReconstruction`](super::SfmrReconstruction), the half a point never
/// belongs to.
///
/// Everything here is addressed by image index, so an operation that adds,
/// drops or reorders images works on this set and an operation that only
/// moves points leaves it alone. The columns are parallel to `images`, except
/// `cameras` and `per_image_adjustments`, which `SfmrImage::camera_index`
/// indexes into, and `rig_frame_data`, which groups images into rig frames.
#[derive(Clone)]
pub struct ImageSet {
    /// Camera intrinsic parameters, indexed by `SfmrImage::camera_index`.
    pub cameras: Vec<CameraIntrinsics>,
    /// Per camera, which parts of its intrinsics vary per image, parallel to
    /// `cameras`; or empty when no camera varies anything, the way an empty
    /// `BundleAdjustOptions::releases` means every camera held. An edit that
    /// adds or removes cameras keeps it in step with `cameras`.
    ///
    /// The adjustments themselves are on the images,
    /// [`SfmrImage::intrinsic_adjustment`]; these flags say which of them a
    /// save writes as solved values, the cameras' `per_image_adjustments` key.
    pub per_image_adjustments: Vec<PerImageAdjustments>,
    /// Registered images with poses.
    pub images: Vec<SfmrImage>,
    /// `(N, THUMBNAIL_SIZE, THUMBNAIL_SIZE, 3)` RGB thumbnails of the source
    /// images (see [`crate::THUMBNAIL_SIZE`]), or `None` for a reconstruction
    /// without them. Whole or absent: row `i` is a downscale of the photograph
    /// `images[i]` names, and nothing fills the column in on load.
    ///
    /// Behind an [`Arc`] because it is one of the two columns that dominate a
    /// reconstruction's memory, so two reconstructions that agree on their
    /// thumbnails share one copy. Nothing writes through it: a producer that
    /// changes the thumbnails builds a new array and wraps it.
    pub thumbnails_y_x_rgb: Option<Arc<Array4<u8>>>,
    /// Per-image depth statistics.
    pub depth_statistics: DepthStatistics,
    /// Depth histogram counts: `depth_histogram_counts[i]` has
    /// `num_histogram_buckets` entries.
    pub depth_histogram_counts: Vec<Vec<u32>>,
    /// Rig definitions and frame groupings. `None` when no multi-camera rigs.
    pub rig_frame_data: Option<RigFrameData>,
}

impl ImageSet {
    /// Number of registered images.
    pub fn image_count(&self) -> usize {
        self.images.len()
    }

    /// Number of camera models.
    pub fn camera_count(&self) -> usize {
        self.cameras.len()
    }

    /// The shared camera of image `image_index`, without the image's intrinsic
    /// adjustment. A caller that projects through the image reads
    /// [`Self::image_intrinsics`] instead.
    pub fn camera_for_image(&self, image_index: usize) -> &CameraIntrinsics {
        &self.cameras[self.images[image_index].camera_index as usize]
    }

    /// The intrinsics image `image_index` was taken with: its camera's, with
    /// the image's intrinsic adjustment applied. Borrowed when the adjustment
    /// is all zero, which is every image of a file without adjustments.
    ///
    /// # Panics
    ///
    /// Panics when `image_index` is past the images, or the image's camera
    /// index past the cameras.
    pub fn image_intrinsics(&self, image_index: usize) -> Cow<'_, CameraIntrinsics> {
        let image = &self.images[image_index];
        let camera = &self.cameras[image.camera_index as usize];
        if image.intrinsic_adjustment.is_zero() {
            Cow::Borrowed(camera)
        } else {
            Cow::Owned(camera.with_adjustment(&image.intrinsic_adjustment))
        }
    }

    /// [`Self::image_intrinsics`] of every image, in image order: what a loop
    /// over observations of many images reads, so that an adjusted camera is
    /// built once per image rather than once per observation.
    pub fn all_image_intrinsics(&self) -> Vec<Cow<'_, CameraIntrinsics>> {
        (0..self.images.len())
            .map(|i| self.image_intrinsics(i))
            .collect()
    }

    /// Every image's intrinsics as a camera list and a per-image index into
    /// it, for a routine that takes a camera list and a per-image camera
    /// column. Borrows the cameras when no image has a non-zero adjustment.
    ///
    /// # Panics
    ///
    /// Panics when an image with a non-zero adjustment names a camera past the
    /// cameras.
    pub fn image_intrinsics_list(&self) -> ImageIntrinsicsList<'_> {
        let image_camera: Vec<u32> = self.images.iter().map(|i| i.camera_index).collect();
        let adjustments: Vec<IntrinsicAdjustment> =
            self.images.iter().map(|i| i.intrinsic_adjustment).collect();
        ImageIntrinsicsList::new(&self.cameras, image_camera, &adjustments)
    }

    /// Which parts of camera `camera_index`'s intrinsics vary per image:
    /// nothing when [`Self::per_image_adjustments`] is empty or does not reach
    /// that camera.
    pub fn camera_per_image_adjustments(&self, camera_index: usize) -> PerImageAdjustments {
        self.per_image_adjustments
            .get(camera_index)
            .copied()
            .unwrap_or_default()
    }

    /// The unit world-space ray `pixel` states in image `image_index`, or
    /// `None` when the index is past the set or the camera model has no ray
    /// there.
    ///
    /// The pose and the image's intrinsics ([`Self::image_intrinsics`]) are
    /// both this set's, so every edit that
    /// triangulates a track -- a shortened one, or one whose camera moved --
    /// assembles its rays here rather than each spelling the pixel-to-world
    /// chain out again.
    pub fn world_ray(&self, image_index: usize, pixel: [f64; 2]) -> Option<nalgebra::Vector3<f64>> {
        let image = self.images.get(image_index)?;
        let camera = self.cameras.get(image.camera_index as usize)?;
        let camera = if image.intrinsic_adjustment.is_zero() {
            Cow::Borrowed(camera)
        } else {
            Cow::Owned(camera.with_adjustment(&image.intrinsic_adjustment))
        };
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
    /// The reference is the mean of **every** posed camera centre in the set
    /// rather than of the ones that see any particular point, because that is
    /// the reference
    /// [`SfmrReconstruction::materialize_points_at_infinity`](super::SfmrReconstruction::materialize_points_at_infinity)
    /// places a bearing at. Every edit that resizes a patch frame measures from
    /// here, so their conversions cancel exactly rather than to within which
    /// images happen to be in a track.
    ///
    /// Each call sums the camera centres again. A caller measuring many points
    /// against one set takes [`Self::camera_centroid`] once and measures each
    /// point with [`Self::placement_scale_from`], which gives the same number.
    pub fn placement_scale(&self, position: &nalgebra::Point3<f64>) -> f64 {
        Self::placement_scale_from(&self.camera_centroid(), position)
    }

    /// The mean of the camera centres of the set's posed images, the
    /// reference [`Self::placement_scale`] measures from; the origin for a
    /// set with no posed image.
    ///
    /// An unposed image, one whose stored rotation or translation is not
    /// finite, has no centre. Counting its placeholder would make the centroid
    /// `NaN`, and every distance measured from it with it.
    pub fn camera_centroid(&self) -> nalgebra::Point3<f64> {
        let mut centroid = nalgebra::Vector3::zeros();
        let mut posed = 0usize;
        for image in &self.images {
            if is_posed(&image.quaternion_wxyz, &image.translation_xyz) {
                centroid += image.camera_center().coords;
                posed += 1;
            }
        }
        if posed > 0 {
            centroid /= posed as f64;
        }
        nalgebra::Point3::from(centroid)
    }

    /// The placement distance of `position` from a `centroid` that
    /// [`Self::camera_centroid`] gave: [`Self::placement_scale`] with the
    /// centroid computed once by the caller, so measuring every point of a
    /// value costs one pass over the images rather than one per point.
    pub fn placement_scale_from(
        centroid: &nalgebra::Point3<f64>,
        position: &nalgebra::Point3<f64>,
    ) -> f64 {
        (position.coords - centroid.coords).norm()
    }
}
