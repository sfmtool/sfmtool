// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Add one observation to a track from a pixel.
//!
//! `specs/core/reconstruction/add-observation.md` is the design. The function
//! here is pure: an [`EditedReconstruction`] and the decoded views go in, a new
//! [`EditedReconstruction`] and a report come out, and the base behind the
//! input's `Arc` is untouched.

use nalgebra::{Point3, Vector3};

use super::edited::{EditError, EditedReconstruction, PointRecord, RecordObservation};
use super::triangulation::triangulate_batch;
use crate::patch::cloud::OrientedPatch;
use crate::patch::keypoint_localize::{localize_patch_keypoints, KeypointLocalizeParams};
use crate::patch::keypoint_subpixel::{refine_patch_keypoints, KeypointSubpixelParams};
use crate::patch::normal_refine::ProjectedImage;

/// Why an observation could not be added. Every variant names what did not
/// hold, because the caller is a menu entry that has to say so in one sentence.
#[derive(Debug, Clone, PartialEq)]
pub enum AddObservationError {
    /// The base's observations are `.sift` feature indexes. An observation
    /// placed at a clicked pixel has no feature behind it, so this edit is
    /// defined on `embedded_patches` reconstructions only.
    NotEmbeddedPatches,
    /// The edited index names no live point.
    NoSuchPoint(u32),
    /// The image index is past the base's image table.
    ImageOutOfRange {
        /// The index named.
        image: u32,
        /// How many images the base holds.
        image_count: usize,
    },
    /// The image already observes this point. A track holds one sighting per
    /// image, and moving an existing one is a different edit.
    ImageAlreadyInTrack(u32),
    /// The pixel is outside the image's own sensor rectangle.
    PixelOutsideImage {
        /// The pixel named, in source-image px.
        pixel: [f32; 2],
        /// The image's `(width, height)`.
        size: (u32, u32),
    },
    /// The point carries no patch frame, so there is nothing to fit against.
    NoPatchFrame(u32),
    /// The point is at infinity: a direction, which no pixel re-triangulates.
    PointAtInfinity(u32),
    /// Fewer decoded views were supplied than the base has images.
    ViewsMissing {
        /// How many were supplied.
        got: usize,
        /// How many the base holds.
        expected: usize,
    },
    /// The localizer dropped the new view: its own gates -- the shift bound,
    /// the grazing cutoff, the consensus floors -- refused it.
    LocalizationRefused(u32),
    /// The new view survived localization but scored below the acceptance bar.
    BelowAcceptanceBar {
        /// The leave-one-out ZNCC the fit reached.
        zncc: f64,
        /// The bar it had to clear.
        bar: f64,
    },
    /// The track including the new observation does not re-triangulate: the
    /// depth is unobservable, or the solve puts the point behind a camera.
    Triangulation,
    /// The record the edit built was refused by the overlay.
    Edit(EditError),
}

impl std::fmt::Display for AddObservationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AddObservationError::NotEmbeddedPatches => write!(
                f,
                "adding an observation from a pixel needs an embedded_patches \
                 reconstruction; this one's observations are .sift features"
            ),
            AddObservationError::NoSuchPoint(i) => write!(f, "no live point at index {i}"),
            AddObservationError::ImageOutOfRange { image, image_count } => write!(
                f,
                "image {image} is past the {image_count} images of the reconstruction"
            ),
            AddObservationError::ImageAlreadyInTrack(i) => {
                write!(f, "image {i} already observes this point")
            }
            AddObservationError::PixelOutsideImage { pixel, size } => write!(
                f,
                "pixel ({}, {}) is outside the {}x{} image",
                pixel[0], pixel[1], size.0, size.1
            ),
            AddObservationError::NoPatchFrame(i) => {
                write!(f, "point {i} carries no patch frame to fit against")
            }
            AddObservationError::PointAtInfinity(i) => {
                write!(
                    f,
                    "point {i} is at infinity, and a direction has no depth to fit"
                )
            }
            AddObservationError::ViewsMissing { got, expected } => write!(
                f,
                "{got} decoded views were supplied and the reconstruction has {expected} images"
            ),
            AddObservationError::LocalizationRefused(i) => write!(
                f,
                "the photometric fit dropped image {i}: the patch does not register there"
            ),
            AddObservationError::BelowAcceptanceBar { zncc, bar } => write!(
                f,
                "the photometric fit scored {zncc:.3}, below the {bar:.3} bar"
            ),
            AddObservationError::Triangulation => write!(
                f,
                "the track does not re-triangulate with the new observation in it"
            ),
            AddObservationError::Edit(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for AddObservationError {}

impl From<EditError> for AddObservationError {
    fn from(e: EditError) -> Self {
        AddObservationError::Edit(e)
    }
}

/// What the fit and the re-triangulation are allowed to do.
///
/// The localization parameters default to the ones `sfm embed-patches` runs
/// with, so an observation added here is placed by the same kernel, at the same
/// settings, as every observation the file already holds.
#[derive(Debug, Clone)]
pub struct AddObservationOptions {
    /// The discrete photometric localization kernel's parameters.
    pub localize: KeypointLocalizeParams,
    /// The sub-pixel stage's parameters, which the embed pass chains after the
    /// discrete one.
    pub refine: KeypointSubpixelParams,
    /// The leave-one-out ZNCC the new view has to reach. Defaults to the
    /// localizer's own absolute floor, which is the bar the embed pass keeps an
    /// observation on.
    pub min_zncc: f64,
}

impl Default for AddObservationOptions {
    fn default() -> Self {
        let localize = KeypointLocalizeParams::default();
        let min_zncc = localize.min_absolute_zncc;
        Self {
            localize,
            refine: KeypointSubpixelParams::default(),
            min_zncc,
        }
    }
}

/// What one added observation did.
#[derive(Debug, Clone, PartialEq)]
pub struct AddObservationReport {
    /// The index the point took in the returned value. The point it names is
    /// the same point: the edit is a modification, and `replaces` records the
    /// index it came from.
    pub point: u32,
    /// The index the point held in the input value.
    pub replaced: u32,
    /// The image the observation was added in.
    pub image: u32,
    /// The pixel the caller named.
    pub clicked_pixel: [f32; 2],
    /// Where the photometric fit put the keypoint, in source-image px.
    pub keypoint: [f32; 2],
    /// How far the fit moved the keypoint from the clicked pixel, in px.
    pub shift_px: f64,
    /// The new view's leave-one-out ZNCC against the rest of the track.
    pub zncc: f64,
    /// How many observations the track holds now.
    pub observation_count: usize,
    /// How far the re-triangulated position moved, in the reconstruction's own
    /// units.
    pub position_shift: f64,
    /// The re-triangulation's condition number.
    pub condition_number: f64,
}

/// Add an observation of `point` in `image`, at `pixel`, to `edited`.
///
/// The clicked pixel is a seed, not the answer: the point's stored patch is
/// registered into `image` by [`localize_patch_keypoints`], seeded there at
/// `pixel` and in every view the track already holds at that view's stored
/// keypoint, and the keypoint the kernel reports for `image` is the observation.
/// The track is then re-triangulated from all of its observations, the new one
/// included.
///
/// The point's colour, normal, patch frame, patch bitmap, stored error and
/// constraint are carried over untouched: this edit places one sighting against
/// the patch as it stands, and moves no observation the track already had, so
/// nothing the frame and the bitmap were fused from has moved.
///
/// `views` is one [`ProjectedImage`] per image of the base, indexed by image
/// index -- the decoded pixels the photometric fit needs, which a
/// reconstruction value does not carry.
///
/// The base behind `edited`'s `Arc` is not written: the returned value shares
/// it, and the point is delete-and-re-added into the overlay.
///
/// # Example
///
/// ```no_run
/// # use std::sync::Arc;
/// # use sfmtool_core::{EditedReconstruction, SfmrReconstruction};
/// # use sfmtool_core::reconstruction::add_observation::{
/// #     add_observation, AddObservationOptions,
/// # };
/// # fn run(base: Arc<SfmrReconstruction>, views: &[sfmtool_core::patch::normal_refine::ProjectedImage<'_>])
/// # -> Result<(), Box<dyn std::error::Error>> {
/// let edited = EditedReconstruction::new(base);
/// let (next, report) = add_observation(
///     &edited,
///     42,
///     7,
///     [120.5, 88.25],
///     views,
///     &AddObservationOptions::default(),
/// )?;
/// assert_eq!(report.replaced, 42);
/// assert!(Arc::ptr_eq(&edited.base, &next.base));
/// # Ok(())
/// # }
/// ```
pub fn add_observation(
    edited: &EditedReconstruction,
    point: u32,
    image: u32,
    pixel: [f32; 2],
    views: &[ProjectedImage<'_>],
    options: &AddObservationOptions,
) -> Result<(EditedReconstruction, AddObservationReport), AddObservationError> {
    // An observation placed at a pixel has a keypoint and no feature index, so
    // this refuses before it reads anything else.
    if edited.has_feature_indexes() {
        return Err(AddObservationError::NotEmbeddedPatches);
    }
    let image_count = edited.image_count();
    if image as usize >= image_count {
        return Err(AddObservationError::ImageOutOfRange { image, image_count });
    }
    if views.len() < image_count {
        return Err(AddObservationError::ViewsMissing {
            got: views.len(),
            expected: image_count,
        });
    }
    let view = edited
        .point(point)
        .ok_or(AddObservationError::NoSuchPoint(point))?;
    if view.observations().iter().any(|o| o.image_index == image) {
        return Err(AddObservationError::ImageAlreadyInTrack(image));
    }
    let camera = views[image as usize].camera;
    let (w, h) = (camera.width, camera.height);
    if !(pixel[0] >= 0.0
        && pixel[1] >= 0.0
        && (pixel[0] as f64) < w as f64
        && (pixel[1] as f64) < h as f64)
    {
        return Err(AddObservationError::PixelOutsideImage {
            pixel,
            size: (w, h),
        });
    }
    let point3d = view.point().clone();
    if point3d.is_at_infinity() {
        return Err(AddObservationError::PointAtInfinity(point));
    }

    // ── The patch the fit registers against: the point's stored frame ──
    let (Some(u), Some(v)) = (view.patch_u_halfvec(), view.patch_v_halfvec()) else {
        return Err(AddObservationError::NoPatchFrame(point));
    };
    let u = Vector3::new(u[0] as f64, u[1] as f64, u[2] as f64);
    let v = Vector3::new(v[0] as f64, v[1] as f64, v[2] as f64);
    let (hu, hv) = (u.norm(), v.norm());
    if !(hu > 0.0 && hv > 0.0) {
        return Err(AddObservationError::NoPatchFrame(point));
    }
    let patch = OrientedPatch::new(point3d.position, u / hu, v / hv, [hu, hv]);

    // ── The view set: the track, then the new image ──
    //
    // The new view is seeded at the clicked pixel and every other at its stored
    // keypoint, which is what gives the kernel a consensus to score the new one
    // against. A single-view localization reports no leave-one-out score at all.
    let track = view.observations();
    let mut view_set: Vec<u32> = track.iter().map(|o| o.image_index).collect();
    let mut seeds: Vec<Option<[f64; 2]>> = (0..track.len())
        .map(|k| view.keypoint_xy(k).map(|p| [p[0] as f64, p[1] as f64]))
        .collect();
    view_set.push(image);
    seeds.push(Some([pixel[0] as f64, pixel[1] as f64]));

    let localized =
        localize_patch_keypoints(&patch, views, &view_set, Some(&seeds), &options.localize);
    let slot = localized
        .views
        .iter()
        .position(|&i| i == image)
        .ok_or(AddObservationError::LocalizationRefused(image))?;
    let zncc = localized.loo_zncc[slot];
    // A `NaN` score is one no round ever produced, and it clears no bar.
    if zncc.is_nan() || zncc < options.min_zncc {
        return Err(AddObservationError::BelowAcceptanceBar {
            zncc,
            bar: options.min_zncc,
        });
    }
    // The sub-pixel stage, seeded at the discrete answer, exactly as the embed
    // pass chains the two. Only the new view's keypoint is read out of it: the
    // observations the track already had keep their stored pixels, so this edit
    // moves one sighting and no other.
    let refined = refine_patch_keypoints(
        &patch,
        views,
        &localized.views,
        Some(
            &localized
                .keypoints
                .iter()
                .map(|&k| Some(k))
                .collect::<Vec<_>>(),
        ),
        &options.refine,
    );
    let fitted = refined
        .views
        .iter()
        .position(|&i| i == image)
        .map_or(localized.keypoints[slot], |k| refined.keypoints[k]);
    let keypoint = [fitted[0] as f32, fitted[1] as f32];
    let shift_px = (fitted[0] - pixel[0] as f64).hypot(fitted[1] - pixel[1] as f64);

    // ── The record: the track with the new sighting in image order ──
    let mut record: PointRecord = view.to_record();
    let at = record
        .observations
        .partition_point(|o| o.image_index < image);
    record.observations.insert(
        at,
        RecordObservation {
            image_index: image,
            feature_index: None,
            keypoint_xy: Some(keypoint),
            // The base carries the column or it does not; when it does, the
            // sighting's confidence is the score the fit reached, in the
            // column's own byte scale.
            confidence: edited
                .has_observation_confidence()
                .then(|| (zncc.clamp(0.0, 1.0) * 255.0).round() as u8),
        },
    );

    // ── Re-triangulation from every observation, the new one included ──
    let mut dirs: Vec<Vector3<f64>> = Vec::with_capacity(record.observations.len());
    let mut centers: Vec<Point3<f64>> = Vec::with_capacity(record.observations.len());
    for obs in &record.observations {
        let v = &views[obs.image_index as usize];
        let [x, y] = obs
            .keypoint_xy
            .expect("an embedded_patches record has keypoints");
        let ray = v.camera.pixel_to_ray(x as f64, y as f64);
        // Camera-to-world carries the canonical (-Z forward) ray into the world
        // frame the triangulator solves in, exactly as the patch spawn does.
        let rot = v.cam_from_world.to_rotation_matrix();
        dirs.push(rot.transpose() * Vector3::new(ray[0], ray[1], ray[2]));
        centers.push(v.cam_from_world.inverse_translation_origin());
    }
    let offsets = [0usize, dirs.len()];
    let tri = triangulate_batch(&dirs, &centers, &offsets)[0];
    if !tri.point.coords.iter().all(|c| c.is_finite())
        || !tri.condition_number.is_finite()
        || !tri.in_front_of_all_cameras
    {
        return Err(AddObservationError::Triangulation);
    }
    let position_shift = (tri.point - point3d.position).norm();
    record.point.position = tri.point;

    let mut next = edited.clone();
    let new_index = next.replace_point(point, record)?;
    let observation_count = next
        .point(new_index)
        .expect("just added")
        .observations()
        .len();
    Ok((
        next,
        AddObservationReport {
            point: new_index,
            replaced: point,
            image,
            clicked_pixel: pixel,
            keypoint,
            shift_px,
            zncc,
            observation_count,
            position_shift,
            condition_number: tri.condition_number,
        },
    ))
}

#[cfg(test)]
mod tests;
