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
use super::triangulation::{triangulate_batch, Triangulation};
use crate::patch::cloud::OrientedPatch;
use crate::patch::keypoint_localize::{localize_patch_keypoints, KeypointLocalizeParams};
use crate::patch::keypoint_subpixel::{refine_patch_keypoints, KeypointSubpixelParams};
use crate::patch::normal_refine::ProjectedImage;
use crate::progress::Progress;
use crate::progress_note;

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
    /// units. Zero for a point that was at infinity, which had no position to
    /// move from.
    pub position_shift: f64,
    /// Whether the point was at infinity and this observation made it finite.
    pub from_infinity: bool,
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
/// `progress` is where this call names its two kernel stages, `localize` and
/// `refine`, the discrete photometric registration and the sub-pixel stage
/// chained after it, each saying how many views it ran over. The stages inside
/// those two kernels are not phases of their own: a call registers one patch
/// over a handful of views, and every other caller runs them once per point
/// inside a rayon loop, where a phase per call would be timing per item. Pass
/// `&Progress::none()` to report nothing, which is one branch per report and
/// no behaviour change at all.
///
/// # Example
///
/// ```no_run
/// # use std::sync::Arc;
/// # use sfmtool_core::progress::Progress;
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
///     &Progress::none(),
/// )?;
/// assert_eq!(report.replaced, 42);
/// assert!(Arc::ptr_eq(&edited.base, &next.base));
/// # Ok(())
/// # }
/// ```
#[allow(clippy::too_many_arguments)]
pub fn add_observation(
    edited: &EditedReconstruction,
    point: u32,
    image: u32,
    pixel: [f32; 2],
    views: &[ProjectedImage<'_>],
    options: &AddObservationOptions,
    progress: &Progress<'_>,
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
    let at_infinity = point3d.is_at_infinity();

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
    let mut patch = OrientedPatch::new(point3d.position, u / hu, v / hv, [hu, hv]);
    // A `w = 0` point's frame is tangent to the direction sphere and its corners
    // are directions, so the renderer has to be told which kind it is; the
    // stored columns are the same two half-vectors either way.
    patch.w = if at_infinity { 0.0 } else { 1.0 };

    // The view set the fit registers over: the track's images seeded at their
    // stored keypoints, then the new one seeded at the click.
    let track = view.observations();
    let mut view_set: Vec<u32> = track.iter().map(|o| o.image_index).collect();
    let mut seeds: Vec<Option<[f64; 2]>> = (0..track.len())
        .map(|k| view.keypoint_xy(k).map(|p| [p[0] as f64, p[1] as f64]))
        .collect();
    view_set.push(image);
    seeds.push(Some([pixel[0] as f64, pixel[1] as f64]));

    // ── The record: the track with the new sighting in image order ──
    //
    // Built with the clicked pixel in it, so the triangulations below read one
    // track rather than remembering the new sighting in a second place; the
    // keypoint and the confidence are written into it once the fit has run.
    let mut record: PointRecord = view.to_record();
    let at = record
        .observations
        .partition_point(|o| o.image_index < image);
    record.observations.insert(
        at,
        RecordObservation {
            image_index: image,
            feature_index: None,
            keypoint_xy: Some(pixel),
            confidence: edited.has_observation_confidence().then_some(0),
        },
    );

    // ── The patch the fit registers against ──
    //
    // A point that has a position registers against the frame it stands on. A
    // point at **infinity** cannot: the fit anchors every view at the point's
    // own projection, and a bearing projects into a second camera as the ray
    // parallel to it rather than as the place the surface is, so under any
    // parallax the search would pull the sighting back onto the bearing's
    // projection and undo the very depth the click supplies. So the click is
    // first used for a **provisional triangulation**, and the fit then runs
    // against the finite patch that gives -- a frame standing at a depth, whose
    // projection in both views lands on the surface. Two passes, and the second
    // is the ordinary finite-point path.
    let fit_patch = if at_infinity {
        let provisional = triangulate_record(&record, views)?.point;
        let scale = placement_scale(&provisional, views);
        if !(scale.is_finite() && scale > 0.0) {
            return Err(AddObservationError::Triangulation);
        }
        OrientedPatch::new(provisional, u / hu, v / hv, [hu * scale, hv * scale])
    } else {
        patch.clone()
    };

    let (keypoint, zncc, shift_px) = fit_keypoint(
        &fit_patch, image, pixel, views, &view_set, &seeds, options, progress,
    )?;

    let fitted = &mut record.observations[at];
    fitted.keypoint_xy = Some(keypoint);
    // The base carries the column or it does not; when it does, the sighting's
    // confidence is the score the fit reached, in the column's own byte scale.
    fitted.confidence = edited
        .has_observation_confidence()
        .then(|| (zncc.clamp(0.0, 1.0) * 255.0).round() as u8);

    // ── Re-triangulation from every observation, the fitted one included ──
    let tri = triangulate_record(&record, views)?;
    // A point that was at infinity becomes finite here: the second bearing is
    // what gives the track a depth, so the record crosses the boundary and its
    // angular patch frame becomes a world-unit one, at the depth this final
    // solve found rather than the provisional one the fit ran against.
    let position_shift = if at_infinity {
        promote_from_infinity(&mut record, &tri.point, views);
        0.0
    } else {
        (tri.point - point3d.position).norm()
    };
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
            from_infinity: at_infinity,
            condition_number: tri.condition_number,
        },
    ))
}

/// Where the photometric fit puts the observation of `point` in `image`, the
/// leave-one-out ZNCC it scored, and how far it moved off `pixel`.
///
/// The **discrete** stage is [`localize_patch_keypoints`] over the track's
/// images followed by the new one, every existing view seeded at its stored
/// keypoint and the new one at the clicked pixel, which is what gives the new
/// view a consensus to be scored against. The **sub-pixel** stage is
/// [`refine_patch_keypoints`], seeded at the discrete answer, and only the new
/// view's keypoint is read out of it: this edit places one sighting and moves
/// none.
///
/// The two stages are the two phases `progress` reports: they are the whole of
/// what this function does, and they are what an added observation's time goes
/// on.
#[allow(clippy::too_many_arguments)]
fn fit_keypoint(
    patch: &OrientedPatch,
    image: u32,
    pixel: [f32; 2],
    views: &[ProjectedImage<'_>],
    view_set: &[u32],
    seeds: &[Option<[f64; 2]>],
    options: &AddObservationOptions,
    progress: &Progress<'_>,
) -> Result<([f32; 2], f64, f64), AddObservationError> {
    let localized = {
        let mut phase = progress.phase("localize");
        let localized =
            localize_patch_keypoints(patch, views, view_set, Some(seeds), &options.localize);
        progress_note!(phase, "{} views", localized.views.len());
        localized
    };
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
    let refined = {
        let mut phase = progress.phase("refine");
        let refined = refine_patch_keypoints(
            patch,
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
        progress_note!(phase, "{} views", refined.views.len());
        refined
    };
    let fitted = refined
        .views
        .iter()
        .position(|&i| i == image)
        .map_or(localized.keypoints[slot], |k| refined.keypoints[k]);
    Ok((
        [fitted[0] as f32, fitted[1] as f32],
        zncc,
        (fitted[0] - pixel[0] as f64).hypot(fitted[1] - pixel[1] as f64),
    ))
}

/// Triangulate `record`'s whole track from the keypoints it holds.
///
/// The rays are built by walking the record rather than the value, so the solve
/// sees exactly the track the edit is about to store and there is no second
/// place where the new sighting has to be remembered. Refused on the three
/// signals the patch spawn refuses on: a non-finite position, an infinite
/// condition number (the depth is not observable) or a solution behind one of
/// the cameras that see it.
fn triangulate_record(
    record: &PointRecord,
    views: &[ProjectedImage<'_>],
) -> Result<Triangulation, AddObservationError> {
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
    Ok(tri)
}

/// The distance an angular patch extent is multiplied by to become a world one
/// at `position`: the distance from the camera-cloud centroid, which is the
/// reference `SfmrReconstruction::materialize_points_at_infinity` measures its
/// own placement from.
fn placement_scale(position: &Point3<f64>, views: &[ProjectedImage<'_>]) -> f64 {
    let mut centroid = Vector3::zeros();
    for view in views {
        centroid += view.cam_from_world.inverse_translation_origin().coords;
    }
    if !views.is_empty() {
        centroid /= views.len() as f64;
    }
    (position.coords - centroid).norm()
}

/// Carry `record`, which is at infinity, across to the finite point `position`.
///
/// `w` becomes 1, and the patch frame is resized at the depth the
/// triangulation just found: the stored half-vectors are angular extents
/// tangent to the direction sphere, so multiplying them by the placement
/// distance is what keeps the patch the apparent size it had -- the same
/// rescale `SfmrReconstruction::materialize_points_at_infinity` applies, and
/// measured from the same reference, the camera-cloud centroid. Leaving them
/// alone would leave a world-unit patch the size of a radian on a point metres
/// away.
///
/// The bitmap is kept: it is the appearance the observation was accepted for
/// agreeing with, and resizing the frame does not change what the tile shows.
/// The normal becomes the resized frame's own, which is the fronto-parallel
/// surfel the tangent frame turns into.
fn promote_from_infinity(
    record: &mut PointRecord,
    position: &Point3<f64>,
    views: &[ProjectedImage<'_>],
) {
    record.point.w = 1.0;
    let scale = placement_scale(position, views);
    if !(scale.is_finite() && scale > 0.0) {
        return;
    }
    for halfvec in [&mut record.patch_u_halfvec, &mut record.patch_v_halfvec] {
        if let Some(h) = halfvec.as_mut() {
            for c in h.iter_mut() {
                *c = (f64::from(*c) * scale) as f32;
            }
        }
    }
    if let (Some(u), Some(v)) = (record.patch_u_halfvec, record.patch_v_halfvec) {
        let u = Vector3::new(u[0] as f64, u[1] as f64, u[2] as f64);
        let v = Vector3::new(v[0] as f64, v[1] as f64, v[2] as f64);
        let n = u.cross(&v);
        if n.norm() > 0.0 {
            let n = n.normalize();
            record.point.normal = Vector3::new(n.x as f32, n.y as f32, n.z as f32);
        }
    }
}

#[cfg(test)]
pub(crate) mod tests;
