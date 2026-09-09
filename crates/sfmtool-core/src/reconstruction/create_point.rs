// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Create a 3D point at infinity from one pixel.
//!
//! `specs/core/reconstruction/create-point.md` is the design. The function here
//! is pure: an [`EditedReconstruction`] and the decoded views go in, a new
//! [`EditedReconstruction`] and a report come out, and the base behind the
//! input's `Arc` is untouched.

use nalgebra::{Point3, Vector3};
use ndarray::Array3;

use super::data::Point3D;
use super::edited::{EditError, EditedReconstruction, PointRecord, RecordObservation};
use crate::camera::remap::{remap_bilinear_mip, sample_bilinear_u8};
use crate::camera::WarpMap;
use crate::patch::cloud::OrientedPatch;
use crate::patch::normal_refine::ProjectedImage;

/// Why a point could not be created. Every variant names what did not hold,
/// because the caller is a menu entry that has to say so in one sentence.
#[derive(Debug, Clone, PartialEq)]
pub enum CreatePointError {
    /// The base's observations are `.sift` feature indexes. An observation
    /// placed at a clicked pixel has no feature behind it, so this edit is
    /// defined on `embedded_patches` reconstructions only.
    NotEmbeddedPatches,
    /// The image index is past the base's image table.
    ImageOutOfRange {
        /// The index named.
        image: u32,
        /// How many images the base holds.
        image_count: usize,
    },
    /// The pixel is outside the image's own sensor rectangle.
    PixelOutsideImage {
        /// The pixel named, in source-image px.
        pixel: [f32; 2],
        /// The image's `(width, height)`.
        size: (u32, u32),
    },
    /// The camera model has no ray for that pixel, or for the pixel a radius
    /// away from it: outside the lens model's valid domain.
    Unprojectable {
        /// The pixel whose ray the model refused.
        pixel: [f32; 2],
    },
    /// The patch radius is not a positive, finite pixel count.
    BadRadius(f32),
    /// Fewer decoded views were supplied than the base has images.
    ViewsMissing {
        /// How many were supplied.
        got: usize,
        /// How many the base holds.
        expected: usize,
    },
    /// The record the edit built was refused by the overlay.
    Edit(EditError),
}

impl std::fmt::Display for CreatePointError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CreatePointError::NotEmbeddedPatches => write!(
                f,
                "creating a point from a pixel needs an embedded_patches \
                 reconstruction; this one's observations are .sift features"
            ),
            CreatePointError::ImageOutOfRange { image, image_count } => write!(
                f,
                "image {image} is past the {image_count} images of the reconstruction"
            ),
            CreatePointError::PixelOutsideImage { pixel, size } => write!(
                f,
                "pixel ({}, {}) is outside the {}x{} image",
                pixel[0], pixel[1], size.0, size.1
            ),
            CreatePointError::Unprojectable { pixel } => write!(
                f,
                "the camera model has no ray for pixel ({}, {})",
                pixel[0], pixel[1]
            ),
            CreatePointError::BadRadius(r) => {
                write!(f, "a patch radius of {r} px is not a positive size")
            }
            CreatePointError::ViewsMissing { got, expected } => write!(
                f,
                "{got} decoded views were supplied and the reconstruction has {expected} images"
            ),
            CreatePointError::Edit(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for CreatePointError {}

impl From<EditError> for CreatePointError {
    fn from(e: EditError) -> Self {
        CreatePointError::Edit(e)
    }
}

/// What the created point carries in the columns the click says nothing about.
#[derive(Debug, Clone)]
pub struct CreatePointOptions {
    /// The confidence written for the one observation, when the base carries
    /// `observation_confidence`. The column's maximum: the sighting is the
    /// user's own, placed where they pointed, and there is no second view to
    /// score it against.
    pub observation_confidence: u8,
    /// The confidence written for the normal, when the base carries
    /// `normal_confidence`. Zero, because a point at infinity carries a zero
    /// normal and the format keeps the two coherent.
    pub normal_confidence: u8,
}

impl Default for CreatePointOptions {
    fn default() -> Self {
        Self {
            observation_confidence: u8::MAX,
            normal_confidence: 0,
        }
    }
}

/// What one created point is.
#[derive(Debug, Clone, PartialEq)]
pub struct CreatePointReport {
    /// The index the created point took in the returned value.
    pub point: u32,
    /// The image the one observation is in.
    pub image: u32,
    /// The pixel the caller named, which is also the observation's keypoint.
    pub pixel: [f32; 2],
    /// The unit world-space direction the point stores.
    pub direction: [f64; 3],
    /// The patch radius the caller named, in this image's pixels.
    pub radius_px: f32,
    /// The frame's angular half-extent, `tan` of the half-angle the radius
    /// subtends, which is what the half-vectors' length is.
    pub half_extent: f64,
    /// The colour sampled at the pixel.
    pub color: [u8; 3],
}

/// The world-space unit ray of `pixel` in `view`, or `None` when the camera
/// model has no ray there.
fn world_ray(view: &ProjectedImage<'_>, pixel: [f64; 2]) -> Option<Vector3<f64>> {
    let ray = view.camera.pixel_to_ray(pixel[0], pixel[1]);
    let cam = Vector3::new(ray[0], ray[1], ray[2]);
    if !cam.iter().all(|c| c.is_finite()) || cam.norm() <= 0.0 {
        return None;
    }
    let rot = view.cam_from_world.to_rotation_matrix();
    Some((rot.transpose() * cam).normalize())
}

/// Create a point at infinity along `pixel`'s ray in `image`, with one
/// observation there.
///
/// One sighting fixes a bearing and no distance, so the point is stored the way
/// the format stores a bearing: `w = 0`, with the pixel's unit world-space ray
/// as its coordinate. Adding a second observation to it re-triangulates it to a
/// finite position; see
/// [`add_observation`](super::add_observation::add_observation).
///
/// `radius_px` is the patch's half-extent in this image's pixels at the clicked
/// pixel. Nothing in a click says how large the point's patch is, so the caller
/// names it, and the stored frame is the angle that many pixels subtend --
/// measured through the camera model, distortion included, so it is the angle
/// the lens actually sees rather than a focal-length approximation.
///
/// `views` is one [`ProjectedImage`] per image of the base, indexed by image
/// index. Only `image`'s entry is read.
///
/// The base behind `edited`'s `Arc` is not written: the returned value shares
/// it, and the point is appended to the overlay's addition set.
///
/// # Example
///
/// ```no_run
/// # use std::sync::Arc;
/// # use sfmtool_core::{EditedReconstruction, SfmrReconstruction};
/// # use sfmtool_core::reconstruction::create_point::{create_point, CreatePointOptions};
/// # fn run(base: Arc<SfmrReconstruction>, views: &[sfmtool_core::patch::normal_refine::ProjectedImage<'_>])
/// # -> Result<(), Box<dyn std::error::Error>> {
/// let edited = EditedReconstruction::new(base);
/// let (next, report) = create_point(
///     &edited,
///     7,                   // the image the user pointed in
///     [120.5, 88.25],      // where they pointed
///     12.0,                // the patch radius, in that image's pixels
///     views,
///     &CreatePointOptions::default(),
/// )?;
/// assert!(Arc::ptr_eq(&edited.base, &next.base));
/// assert_eq!(next.point(report.point).expect("just created").point().w, 0.0);
/// # Ok(())
/// # }
/// ```
pub fn create_point(
    edited: &EditedReconstruction,
    image: u32,
    pixel: [f32; 2],
    radius_px: f32,
    views: &[ProjectedImage<'_>],
    options: &CreatePointOptions,
) -> Result<(EditedReconstruction, CreatePointReport), CreatePointError> {
    // The cheap refusals first, so a caller greying a menu entry gets the same
    // answers without decoding anything.
    if edited.has_feature_indexes() {
        return Err(CreatePointError::NotEmbeddedPatches);
    }
    let image_count = edited.image_count();
    if image as usize >= image_count {
        return Err(CreatePointError::ImageOutOfRange { image, image_count });
    }
    if views.len() < image_count {
        return Err(CreatePointError::ViewsMissing {
            got: views.len(),
            expected: image_count,
        });
    }
    if !(radius_px.is_finite() && radius_px > 0.0) {
        return Err(CreatePointError::BadRadius(radius_px));
    }
    let view = &views[image as usize];
    let (w, h) = (view.camera.width, view.camera.height);
    if !(pixel[0] >= 0.0
        && pixel[1] >= 0.0
        && (pixel[0] as f64) < w as f64
        && (pixel[1] as f64) < h as f64)
    {
        return Err(CreatePointError::PixelOutsideImage {
            pixel,
            size: (w, h),
        });
    }

    // ── The bearing: the pixel's own ray, in world ──
    let px = [pixel[0] as f64, pixel[1] as f64];
    let direction = world_ray(view, px).ok_or(CreatePointError::Unprojectable { pixel })?;

    // ── The frame's size: the angle `radius_px` subtends at this pixel ──
    //
    // Measured by unprojecting a pixel one radius away along each sensor axis
    // and taking the angle to the centre ray, so the lens's own distortion is
    // in the answer. The two axes are averaged into one half-angle because a
    // patch frame is square. `tan` of it is the half-vector length: an infinity
    // patch's corner is the direction `d + s·u`, and with `u ⊥ d` and `|d| = 1`
    // that corner sits at `atan(|u|)` off `d`.
    let r = radius_px as f64;
    let mut angles = [0.0_f64; 2];
    for (axis, angle) in angles.iter_mut().enumerate() {
        // Step whichever way stays on the sensor, so a click at the frame edge
        // still measures a real angle.
        let limit = if axis == 0 { w as f64 } else { h as f64 };
        let step = if px[axis] + r < limit { r } else { -r };
        let mut probe = px;
        probe[axis] += step;
        if !(probe[axis] >= 0.0 && probe[axis] < limit) {
            return Err(CreatePointError::BadRadius(radius_px));
        }
        let other = world_ray(view, probe).ok_or(CreatePointError::Unprojectable {
            pixel: [probe[0] as f32, probe[1] as f32],
        })?;
        *angle = direction.dot(&other).clamp(-1.0, 1.0).acos();
    }
    let half_angle = 0.5 * (angles[0] + angles[1]);
    let half_extent = half_angle.tan();
    if !(half_extent.is_finite() && half_extent > 0.0) {
        return Err(CreatePointError::BadRadius(radius_px));
    }

    // ── The frame: fronto-parallel in this camera, tangent to the bearing ──
    //
    // The tangent-sphere frame the format states for a `w = 0` point, pinned in
    // rotation by the camera's own up axis so the stored patch is the upright
    // crop the user is looking at rather than an arbitrary rotation of it.
    let rot = view.cam_from_world.to_rotation_matrix();
    let up_hint = rot.transpose() * Vector3::y();
    let patch = OrientedPatch::from_infinity_direction(
        Point3::from(direction),
        up_hint,
        [half_extent, half_extent],
    );

    // ── The bitmap: this image, rendered through that frame ──
    let patch_bitmap = edited
        .base
        .point_set
        .patch_bitmaps_y_x_rgba
        .as_ref()
        .map(|b| render_bitmap(&patch, view, b.shape()[1], b.shape()[3]));

    let color = sample_color(view, px);

    let record = PointRecord {
        point: Point3D {
            position: Point3::from(direction),
            w: 0.0,
            color,
            error: 0.0,
            // A point at infinity carries no normal, which is what the demotion
            // pass leaves and what the format states for a `w = 0` row.
            normal: Vector3::zeros(),
        },
        observations: vec![RecordObservation {
            image_index: image,
            feature_index: None,
            keypoint_xy: edited.has_keypoints().then_some(pixel),
            confidence: edited
                .has_observation_confidence()
                .then_some(options.observation_confidence),
        }],
        patch_u_halfvec: edited
            .has_patch_frames()
            .then(|| halfvec(patch.u_axis * half_extent)),
        patch_v_halfvec: edited
            .has_patch_frames()
            .then(|| halfvec(patch.v_axis * half_extent)),
        patch_bitmap,
        normal_confidence: edited
            .has_normal_confidence()
            .then_some(options.normal_confidence),
        // A created point states nothing about a distance a caller owns, so it
        // is free.
        constraint: edited.has_point_constraints().then_some((
            sfmr_format::POINT_CONSTRAINT_FREE,
            f64::NAN,
            sfmr_format::NO_REFERENCE_IMAGE,
        )),
    };

    let mut next = edited.clone();
    let index = next.add_point(record)?;
    Ok((
        next,
        CreatePointReport {
            point: index,
            image,
            pixel,
            direction: [direction.x, direction.y, direction.z],
            radius_px,
            half_extent,
            color,
        },
    ))
}

/// A world half-vector as the column's `f32` triple.
fn halfvec(v: Vector3<f64>) -> [f32; 3] {
    [v.x as f32, v.y as f32, v.z as f32]
}

/// The image's colour at `pixel`, as the point's RGB.
///
/// A grey image gives the same value in all three channels, which is what the
/// column means by a colourless point.
fn sample_color(view: &ProjectedImage<'_>, pixel: [f64; 2]) -> [u8; 3] {
    let level = view.pyramid.level(0);
    let channels = level.channels();
    let mut color = [0u8; 3];
    for (c, out) in color.iter_mut().enumerate() {
        let channel = if channels >= 3 { c as u32 } else { 0 };
        let v = sample_bilinear_u8(level, pixel[0] as f32, pixel[1] as f32, channel);
        *out = v.round().clamp(0.0, 255.0) as u8;
    }
    color
}

/// The `(R, R, C)` patch bitmap: `image` resampled through `patch`'s frame, the
/// way every stored patch bitmap is rendered.
///
/// A pixel the warp cannot sample is left black, and the alpha channel -- the
/// fourth, when the column carries one -- is opaque everywhere, because the
/// whole tile is content this one image saw.
fn render_bitmap(
    patch: &OrientedPatch,
    view: &ProjectedImage<'_>,
    resolution: usize,
    channels: usize,
) -> Array3<u8> {
    let mut map = WarpMap::from_patch(patch, view.camera, view.cam_from_world, resolution as u32);
    map.compute_svd();
    let tile = remap_bilinear_mip(view.pyramid, &map);
    let src_channels = tile.channels();
    let mut out = Array3::<u8>::zeros((resolution, resolution, channels));
    for row in 0..resolution {
        for col in 0..resolution {
            for c in 0..channels {
                out[[row, col, c]] = if c >= 3 {
                    u8::MAX
                } else if src_channels >= 3 {
                    tile.get_pixel(col as u32, row as u32, c as u32)
                } else {
                    tile.get_pixel(col as u32, row as u32, 0)
                };
            }
        }
    }
    out
}

#[cfg(test)]
mod tests;
