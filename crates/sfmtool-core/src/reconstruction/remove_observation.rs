// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Remove one observation from a track.
//!
//! `specs/core/reconstruction/remove-observation.md` is the design. The
//! function here is pure: an [`EditedReconstruction`] goes in, a new
//! [`EditedReconstruction`] and a report come out, and the base behind the
//! input's `Arc` is untouched.

use nalgebra::{Point3, Vector3};

use super::data::ImageTable;
use super::edited::{EditError, EditedReconstruction, PointRecord};
use super::triangulation::{triangulate_batch, Triangulation};

/// Why an observation could not be removed. Every variant names what did not
/// hold, because the caller is a menu entry that has to say so in one sentence.
#[derive(Debug, Clone, PartialEq)]
pub enum RemoveObservationError {
    /// The edited index names no live point.
    NoSuchPoint(u32),
    /// The image index is past the base's image table.
    ImageOutOfRange {
        /// The index named.
        image: u32,
        /// How many images the base holds.
        image_count: usize,
    },
    /// The image does not observe this point, so there is no row to take out.
    ImageNotInTrack(u32),
    /// The track left over does not re-triangulate: the depth is unobservable,
    /// or the solve puts the point behind a camera.
    Triangulation,
    /// The record the edit built was refused by the overlay.
    Edit(EditError),
}

impl std::fmt::Display for RemoveObservationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RemoveObservationError::NoSuchPoint(i) => write!(f, "no live point at index {i}"),
            RemoveObservationError::ImageOutOfRange { image, image_count } => write!(
                f,
                "image {image} is past the {image_count} images of the reconstruction"
            ),
            RemoveObservationError::ImageNotInTrack(i) => {
                write!(f, "image {i} does not observe this point")
            }
            RemoveObservationError::Triangulation => write!(
                f,
                "the track does not re-triangulate with that observation taken out"
            ),
            RemoveObservationError::Edit(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for RemoveObservationError {}

impl From<EditError> for RemoveObservationError {
    fn from(e: EditError) -> Self {
        RemoveObservationError::Edit(e)
    }
}

/// What one removed observation did.
#[derive(Debug, Clone, PartialEq)]
pub struct RemoveObservationReport {
    /// The index the point took in the returned value, or `None` when the
    /// track had one observation and the point was deleted with it.
    pub point: Option<u32>,
    /// The index the point held in the input value.
    pub replaced: u32,
    /// The image whose observation was taken out.
    pub image: u32,
    /// How many observations the track holds now. Zero for a deleted point.
    pub observation_count: usize,
    /// Whether the point was deleted, its last observation having been the one
    /// removed.
    pub deleted: bool,
    /// Whether the point crossed to infinity: one observation is a bearing and
    /// no distance.
    pub to_infinity: bool,
    /// Whether the position was re-solved from the observations that remain.
    pub retriangulated: bool,
    /// Where the point stands now: a world position for a finite point, a unit
    /// world direction for one at infinity, and the position it had for a
    /// deleted one.
    pub position: [f64; 3],
    /// How far the position moved, in the reconstruction's own units. Zero for
    /// a point that was deleted, that crossed to infinity, or whose position
    /// was not re-solved.
    pub position_shift: f64,
    /// The re-triangulation's condition number, or `NaN` when no
    /// re-triangulation ran.
    pub condition_number: f64,
}

/// Remove the observation of `point` in `image` from `edited`.
///
/// The track keeps every other sighting exactly as it stands -- its keypoint,
/// its feature index, its confidence -- and so do the point's colour, normal,
/// patch frame, patch bitmap, stored error and constraint. What the removal
/// changes is the geometry the remaining sightings state, and how much of it
/// they state at all:
///
/// - **Two or more remain**, and the point is finite: it is re-triangulated
///   from them by [`triangulate_batch`], the same solve
///   [`add_observation`](super::add_observation::add_observation) runs.
/// - **One remains**: a single sighting fixes a bearing and no distance, so the
///   point becomes one -- `w = 0`, its coordinate the remaining observation's
///   unit world ray -- and the patch frame is divided by the placement distance
///   it was multiplied by on the way in, which keeps its angular size. This is
///   the mirror of [`create_point`](super::create_point::create_point), whose
///   points are bearings for exactly the same reason.
/// - **None remains**: the point is deleted, and the report says so.
///
/// Unlike adding one, removing a row needs no feature to be invented, so this
/// edit is defined on `sift_files` values as well as `embedded_patches` ones. A
/// track whose observations carry no inline keypoint states no ray, so there
/// the geometry is left exactly as it stands and the report's `retriangulated`
/// is false.
///
/// The base behind `edited`'s `Arc` is not written: the returned value shares
/// it, and the point is delete-and-re-added into the overlay (or, with nothing
/// left of its track, deleted).
///
/// # Example
///
/// ```no_run
/// # use std::sync::Arc;
/// # use sfmtool_core::{EditedReconstruction, SfmrReconstruction};
/// # use sfmtool_core::reconstruction::remove_observation::remove_observation;
/// # fn run(base: Arc<SfmrReconstruction>) -> Result<(), Box<dyn std::error::Error>> {
/// let edited = EditedReconstruction::new(base);
/// let (next, report) = remove_observation(&edited, 42, 7)?;
/// assert_eq!(report.replaced, 42);
/// assert!(Arc::ptr_eq(&edited.base, &next.base));
/// # Ok(())
/// # }
/// ```
pub fn remove_observation(
    edited: &EditedReconstruction,
    point: u32,
    image: u32,
) -> Result<(EditedReconstruction, RemoveObservationReport), RemoveObservationError> {
    let image_count = edited.image_count();
    if image as usize >= image_count {
        return Err(RemoveObservationError::ImageOutOfRange { image, image_count });
    }
    let view = edited
        .point(point)
        .ok_or(RemoveObservationError::NoSuchPoint(point))?;
    if !view.observations().iter().any(|o| o.image_index == image) {
        return Err(RemoveObservationError::ImageNotInTrack(image));
    }

    let was_at_infinity = view.point().is_at_infinity();
    let position_before = view.point().position;
    let mut record: PointRecord = view.to_record();
    let at = record
        .observations
        .iter()
        .position(|o| o.image_index == image)
        .expect("the track was just checked for this image");
    // The row goes out whole: its keypoint, its feature index and its
    // confidence are fields of the entry, so there is no parallel column left
    // holding a value for a sighting that is gone.
    record.observations.remove(at);

    let table = &edited.base.image_table;
    let mut report = RemoveObservationReport {
        point: None,
        replaced: point,
        image,
        observation_count: record.observations.len(),
        deleted: false,
        to_infinity: false,
        retriangulated: false,
        position: [position_before.x, position_before.y, position_before.z],
        position_shift: 0.0,
        condition_number: f64::NAN,
    };

    let mut next = edited.clone();
    match record.observations.len() {
        // Nothing sees the point any more, so there is no point.
        0 => {
            next.delete_point(point)?;
            report.deleted = true;
            return Ok((next, report));
        }
        // One sighting is a bearing and no distance. A point that is already a
        // bearing stays the one it is: it has no depth to give up.
        1 => {
            if !was_at_infinity {
                if let Some(direction) = observation_ray(&record, 0, table) {
                    demote_to_infinity(&mut record, &position_before, &direction, table);
                    report.to_infinity = true;
                    report.position = [direction.x, direction.y, direction.z];
                }
            }
        }
        // Two or more rays state a depth again. A bearing keeps being one: its
        // stored coordinate is a direction, and re-solving it would be a
        // different edit.
        _ => {
            if !was_at_infinity && every_observation_has_a_keypoint(&record) {
                let tri = triangulate_record(&record, table)?;
                record.point.position = tri.point;
                report.retriangulated = true;
                report.position = [tri.point.x, tri.point.y, tri.point.z];
                report.position_shift = (tri.point - position_before).norm();
                report.condition_number = tri.condition_number;
            }
        }
    }

    let new_index = next.replace_point(point, record)?;
    report.point = Some(new_index);
    Ok((next, report))
}

/// Whether every observation of `record` carries an inline keypoint, which is
/// what a ray needs. A `sift_files` track without the optional inline column
/// carries none.
fn every_observation_has_a_keypoint(record: &PointRecord) -> bool {
    record.observations.iter().all(|o| o.keypoint_xy.is_some())
}

/// The unit world-space ray of `record`'s observation `k`, or `None` when it
/// carries no inline keypoint or the camera model has no ray there.
fn observation_ray(
    record: &PointRecord,
    k: usize,
    table: &ImageTable,
) -> Option<nalgebra::Vector3<f64>> {
    let obs = record.observations.get(k)?;
    let [x, y] = obs.keypoint_xy?;
    let image = table.images.get(obs.image_index as usize)?;
    let camera = table.cameras.get(image.camera_index as usize)?;
    let ray = camera.pixel_to_ray(x as f64, y as f64);
    let cam = Vector3::new(ray[0], ray[1], ray[2]);
    if !cam.iter().all(|c| c.is_finite()) || cam.norm() <= 0.0 {
        return None;
    }
    let rot = image.quaternion_wxyz.to_rotation_matrix();
    Some((rot.transpose() * cam).normalize())
}

/// Triangulate `record`'s whole track from the keypoints it holds, over the
/// poses and lenses of `table`.
///
/// The rays are built by walking the record rather than the value, so the solve
/// sees exactly the track the edit is about to store. Refused on the three
/// signals the patch spawn refuses on: a non-finite position, an infinite
/// condition number (the depth is not observable) or a solution behind one of
/// the cameras that see it.
fn triangulate_record(
    record: &PointRecord,
    table: &ImageTable,
) -> Result<Triangulation, RemoveObservationError> {
    let mut dirs: Vec<Vector3<f64>> = Vec::with_capacity(record.observations.len());
    let mut centers: Vec<Point3<f64>> = Vec::with_capacity(record.observations.len());
    for (k, obs) in record.observations.iter().enumerate() {
        let direction =
            observation_ray(record, k, table).ok_or(RemoveObservationError::ImageOutOfRange {
                image: obs.image_index,
                image_count: table.images.len(),
            })?;
        let image = &table.images[obs.image_index as usize];
        dirs.push(direction);
        centers.push(image.camera_center());
    }
    let offsets = [0usize, dirs.len()];
    let tri = triangulate_batch(&dirs, &centers, &offsets)[0];
    if !tri.point.coords.iter().all(|c| c.is_finite())
        || !tri.condition_number.is_finite()
        || !tri.in_front_of_all_cameras
    {
        return Err(RemoveObservationError::Triangulation);
    }
    Ok(tri)
}

/// Carry `record`, which stands at `position`, back to a bearing along
/// `direction`.
///
/// `w` becomes 0 and the coordinate becomes the unit direction, which is what
/// the format states a point at infinity is. The patch frame is **divided** by
/// the placement distance the point stood at, which is the inverse of the
/// resize `add_observation` applies on the way in: the stored half-vectors go
/// back to being angular extents tangent to the direction sphere, and the patch
/// keeps the apparent size it had. Leaving them alone would leave a patch a
/// radian wide on a bearing.
///
/// The bitmap is kept -- it is the appearance the point is known by, and
/// resizing the frame does not change what the tile shows -- and the normal
/// becomes zero, which is what the format states for a `w = 0` row and what a
/// created point carries.
fn demote_to_infinity(
    record: &mut PointRecord,
    position: &Point3<f64>,
    direction: &Vector3<f64>,
    table: &ImageTable,
) {
    record.point.w = 0.0;
    record.point.position = Point3::from(*direction);
    record.point.normal = Vector3::zeros();
    if let Some(confidence) = record.normal_confidence.as_mut() {
        // The format keeps the normal and its confidence coherent, and a zero
        // normal is no statement about a surface.
        *confidence = 0;
    }
    let scale = table.placement_scale(position);
    if !(scale.is_finite() && scale > 0.0) {
        return;
    }
    for halfvec in [&mut record.patch_u_halfvec, &mut record.patch_v_halfvec] {
        if let Some(h) = halfvec.as_mut() {
            for c in h.iter_mut() {
                *c = (f64::from(*c) / scale) as f32;
            }
        }
    }
}

#[cfg(test)]
mod tests;
