// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Each observation's readings on its own `R×R` render, as the `.sfmr` file
//! stores them ([`ObservationReading`]): the render's ZNCC self-similarity
//! ellipse, its viewing angle, tilt direction and zoom, and its plain and
//! blur-matched scores against the point's stored bitmap.
//!
//! [`observation_reading`] builds a row from readings a caller already took on
//! the render, as the bench commit does from its evaluation;
//! [`read_view_tile`] takes them on a rendered tile; and
//! [`read_cloud_observations`] renders and reads every observation of a
//! reconstruction, for a writer that has the photographs. See
//! `specs/formats/sfmr-file-format.md` § "Observation readings".

use std::f64::consts::PI;
use std::sync::atomic::{AtomicUsize, Ordering};

use rayon::prelude::*;

use crate::camera::sampler::SamplerChoice;
use crate::camera::warp_map::singular_values_2x2;
use crate::patch::cloud::PatchCloud;
use crate::patch::member_coherence::MemberCoherenceParams;
use crate::patch::normal_refine::ProjectedImage;
use crate::patch::reference_view::{render_view_tile, ViewTile};
use crate::patch::self_similarity::{
    zncc_self_similarity_radius, PatchTile, SelfSimilarityEllipse, SelfSimilarityParams,
};
use crate::patch::stored_bitmap::{stored_bitmap_planes, BitmapScorer};
use crate::progress::{Cancelled, Progress};
use crate::reconstruction::{ObservationReading, SfmrReconstruction};

/// The angle of an axis, in radians, brought into `[0, π)`; `NaN` stays
/// `NaN`.
fn axis_angle(radians: f64) -> f64 {
    let a = radians.rem_euclid(PI);
    // `rem_euclid` can round a value just under 0 up to π.
    if a >= PI {
        0.0
    } else {
        a
    }
}

/// A grid-frame ellipse angle (from the grid's `x` towards its row-down `y`)
/// as an angle from the patch's `u` axis towards its `v` axis, and back: the
/// grid's `x` runs along `+u` and its `y` along `−v`, so the one is the other
/// reflected, in `[0, π)`.
pub fn grid_angle_to_patch_angle(angle: f64) -> f64 {
    axis_angle(-angle)
}

/// The zoom `[least, most]` of a render whose Jacobian, image px per grid px,
/// is `jacobian`: `[1/σ_major, 1/σ_minor]`, grid px per photograph px.
pub fn zoom_of_jacobian(jacobian: [[f64; 2]; 2]) -> [f64; 2] {
    let [major, minor] = singular_values_2x2(jacobian);
    [1.0 / major, 1.0 / minor]
}

/// The row of readings a caller already took on one observation's render.
///
/// `ellipse` is the render's whole-tile self-similarity ellipse in grid px
/// (`None`, or `NaN` axes, for a render not read, which makes the whole row
/// [`ObservationReading::NOT_MEASURED`]); `viewing_angle_deg` and
/// `tilt_direction_deg` are the render's viewing angle and tilt direction
/// ([`ViewingAngle`](crate::patch::normal_refine::ViewingAngle)); `zoom` is
/// `[least, most]`; `scores` the plain and blur-matched scores against the
/// stored bitmap, `None` where they were not read.
pub fn observation_reading(
    ellipse: Option<&SelfSimilarityEllipse>,
    viewing_angle_deg: Option<f64>,
    tilt_direction_deg: Option<f64>,
    zoom: Option<[f64; 2]>,
    scores: Option<(f64, f64)>,
) -> ObservationReading {
    let Some(ellipse) = ellipse.filter(|e| e.axes.iter().all(|a| !a.is_nan())) else {
        return ObservationReading::NOT_MEASURED;
    };
    let f = |v: Option<f64>| v.map_or(f32::NAN, |v| v as f32);
    let (plain, blur_matched) = scores.unzip();
    ObservationReading {
        ellipse_axes: ellipse.axes.map(|a| a as f32),
        ellipse_axes_is_at_least: ellipse.axes_is_at_least,
        ellipse_major_angle: grid_angle_to_patch_angle(ellipse.major_angle) as f32,
        cos_view_angle: f(viewing_angle_deg.map(|a| a.to_radians().cos())),
        tilt_angle: f(tilt_direction_deg.map(|a| axis_angle(a.to_radians()))),
        zoom: zoom.map_or([f32::NAN; 2], |z| z.map(|v| v as f32)),
        plain_bitmap_zncc: f(plain),
        blur_matched_bitmap_zncc: f(blur_matched),
    }
}

/// The ellipse a stored row records, back in the grid frame
/// (`x` column-right, `y` row-down), with its matrix rebuilt from the axes and
/// the angle: what a reader that shows the row as a reading uses. `None` for a
/// row with nothing measured.
pub fn stored_ellipse(row: &ObservationReading) -> Option<SelfSimilarityEllipse> {
    if !row.is_measured() {
        return None;
    }
    let axes = row.ellipse_axes.map(f64::from);
    let angle = f64::from(row.ellipse_major_angle);
    let grid_angle = if angle.is_nan() {
        f64::NAN
    } else {
        grid_angle_to_patch_angle(angle)
    };
    Some(SelfSimilarityEllipse::from_axes(
        axes,
        row.ellipse_axes_is_at_least,
        grid_angle,
    ))
}

/// The whole-tile self-similarity reading of `tile` with the default
/// parameters, the reading the bench takes of a row's tile.
pub fn view_tile_ellipse(tile: &ViewTile) -> SelfSimilarityEllipse {
    let planes = tile.planes();
    let side = planes.side;
    zncc_self_similarity_radius(
        &PatchTile {
            values: &planes.values,
            channels: planes.channels.min(3),
            width: side,
            height: side,
        },
        Some(&planes.data),
        [0, 0, side, side],
        &SelfSimilarityParams::default(),
    )
    .ellipse
}

/// Read one observation's rendered tile: its self-similarity ellipse, and its
/// scores against the stored bitmap `scorer` holds, `(1, 1)` where `is_reference`
/// says the bitmap is this tile's render, and `NaN` without a scorer.
pub fn read_view_tile(
    tile: &ViewTile,
    scorer: Option<&mut BitmapScorer<'_>>,
    is_reference: bool,
) -> ObservationReading {
    let ellipse = view_tile_ellipse(tile);
    let measured = ellipse.axes.iter().all(|a| !a.is_nan());
    let scores = match scorer {
        _ if !measured => None,
        Some(_) if is_reference => Some((1.0, 1.0)),
        Some(scorer) => {
            let score = scorer.score(&tile.planes(), Some(ellipse.matrix));
            Some((score.plain_zncc, score.blur_matched_zncc))
        }
        None => None,
    };
    observation_reading(
        Some(&ellipse),
        tile.viewing_angle.map(|a| a.angle_deg),
        tile.viewing_angle.and_then(|a| a.tilt_direction_deg),
        tile.jacobian.map(zoom_of_jacobian),
        scores,
    )
}

/// Render and read every observation of every patch of `cloud` at its stored
/// keypoint, parallel across patches (rayon), scoring each against its
/// point's stored bitmap in `recon`.
///
/// Returns one row per observation of `recon`, parallel to its tracks. A row
/// is [`ObservationReading::NOT_MEASURED`] where its point has no patch in
/// `cloud`, its photograph is not to hand (`views[image]` is `None`), or its
/// tile has no data. The scores are `NaN` where the point has no stored bitmap
/// at `resolution` (a zero row, or no bitmap column), and `1` for the point's
/// reference observation ([`PointSet::reference_observations`](crate::PointSet::reference_observations)),
/// whose render the bitmap is. Each tile is rendered as the stored bitmap is,
/// through the patch re-anchored on the keypoint at `resolution` with the
/// sampler `sampler` picks ([`render_view_tile`]), and scored by
/// [`BitmapScorer`] over member coherence's window, as the bench scores a row.
///
/// # Errors
///
/// [`Cancelled`] when `progress` was cancelled before every patch was read.
///
/// # Panics
///
/// Panics if `recon` carries no inline keypoints, or a track's image index is
/// out of range for `views`.
pub fn read_cloud_observations(
    cloud: &PatchCloud,
    recon: &SfmrReconstruction,
    views: &[Option<ProjectedImage<'_>>],
    resolution: usize,
    sampler: SamplerChoice,
    progress: &Progress<'_>,
) -> Result<Vec<ObservationReading>, Cancelled> {
    let keypoints = recon
        .keypoints_xy()
        .expect("read_cloud_observations needs a reconstruction with inline keypoints");
    let set = &recon.point_set;
    let offsets = &set.observation_offsets;
    let bitmaps = set.patch_bitmaps_y_x_rgba.as_deref();
    let references = set.reference_observations.as_deref();
    let window = MemberCoherenceParams::default().window;
    let total = cloud.patches.len();
    let step = (total / 100).max(1);
    let done = AtomicUsize::new(0);
    let read: Vec<(usize, Vec<ObservationReading>)> = cloud
        .patches
        .par_iter()
        .zip(cloud.point_indexes.par_iter())
        .filter_map(|(patch, &pid)| {
            if progress.is_cancelled() {
                return None;
            }
            let p = pid as usize;
            let planes = bitmaps
                .filter(|b| b.shape()[1] == resolution)
                .and_then(|b| stored_bitmap_planes(b.index_axis(ndarray::Axis(0), p)));
            let mut scorer = planes
                .as_ref()
                .map(|planes| BitmapScorer::new(planes, window));
            let reference = references
                .and_then(|r| r.get(p))
                .and_then(|&r| usize::try_from(r).ok());
            let rows = (offsets[p]..offsets[p + 1])
                .enumerate()
                .map(|(k, j)| {
                    let Some(view) = &views[set.tracks[j].image_index as usize] else {
                        return ObservationReading::NOT_MEASURED;
                    };
                    let keypoint = [f64::from(keypoints[[j, 0]]), f64::from(keypoints[[j, 1]])];
                    let tile = render_view_tile(
                        patch,
                        view,
                        Some(keypoint),
                        resolution,
                        sampler,
                        progress,
                    );
                    read_view_tile(&tile, scorer.as_mut(), reference == Some(k))
                })
                .collect();
            let n = done.fetch_add(1, Ordering::Relaxed) + 1;
            if n.is_multiple_of(step) || n == total {
                progress.count(n as u64, Some(total as u64), "patches");
            }
            Some((p, rows))
        })
        .collect();
    progress.check_cancel()?;
    let mut out = vec![ObservationReading::NOT_MEASURED; set.tracks.len()];
    for (p, rows) in read {
        out[offsets[p]..offsets[p + 1]].copy_from_slice(&rows);
    }
    Ok(out)
}
