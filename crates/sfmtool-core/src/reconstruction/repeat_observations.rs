// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Dropping a reconstruction's repeat observations: the observations of a
//! point beyond one per image.
//!
//! A sound track sees its point at most once in each photograph. A GLOMAP
//! solve can hand back a point whose track holds two or more observations in
//! one image, each a different feature of that image. This keeps, in each such
//! image, the observation that agrees best with the point, and drops the
//! others. Nothing moves: the point keeps its position and every image it was
//! seen in, and only its observation list is shorter.
//!
//! See `specs/core/reconstruction/drop-repeat-observations.md` for the design.

use std::collections::BTreeMap;

use crate::progress::Progress;
use crate::progress_info;
use crate::reconstruction::data::{
    observation_reprojection_error, rms_reprojection_error_px, ReconstructionError,
    SfmrReconstruction,
};

/// What [`SfmrReconstruction::drop_repeat_observations`] did.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct RepeatObservationsReport {
    /// Observations dropped, over every point.
    pub observations_dropped: usize,
    /// Points that lost at least one observation.
    pub points_changed: usize,
}

impl SfmrReconstruction {
    /// This value with every repeat observation dropped, and what was dropped.
    ///
    /// For each point and each image in which its track holds more than one
    /// observation, the observation with the lowest reprojection error against
    /// the point is kept and the others in that image are dropped. The error is
    /// the one [`Self::recompute_point_errors`] measures: the point projected
    /// through the image's pose and its own intrinsics, a point at infinity
    /// through the rotation alone, against the observation's pixel. An
    /// observation the camera cannot image (behind it, or outside its lens's
    /// domain), or whose pixel is unknown, ranks after every one that projects.
    /// Equal errors, and a group where none projects, keep the observation that
    /// comes first in the point's track.
    ///
    /// No point moves and no point is removed, and each point keeps every image
    /// it was seen in. Every per-observation column takes the same row
    /// selection, and a reference observation that is dropped becomes `-1`. A
    /// point that lost an observation has its RMS reprojection error scored
    /// again over the observations it keeps. The per-image depth statistics
    /// and histograms are derived again from the shortened tracks; the points'
    /// normals are kept as they are.
    ///
    /// The pixels are the inline `keypoints_xy` column when the value carries
    /// one, and otherwise are read from the `.sift` files of the images the
    /// changed points are seen in. A value with no repeat observation comes
    /// back as it is, content hashes included, and reads no file.
    ///
    /// `progress` is how the call is asked to stop; a cancel is read between
    /// stages. Pass `&Progress::none()` to report nothing and never stop.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use sfmtool_core::progress::Progress;
    /// use sfmtool_core::SfmrReconstruction;
    /// # fn run(recon: &SfmrReconstruction) -> Result<(), Box<dyn std::error::Error>> {
    /// let (cleaned, report) = recon.drop_repeat_observations(&Progress::none())?;
    /// println!(
    ///     "dropped {} observations from {} points",
    ///     report.observations_dropped, report.points_changed
    /// );
    /// # let _ = cleaned;
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    ///
    /// [`ReconstructionError::SiftRead`] when a `.sift` file the pixels come
    /// from cannot be read, and [`ReconstructionError::Cancelled`] when the call
    /// was asked to stop.
    pub fn drop_repeat_observations(
        &self,
        progress: &Progress<'_>,
    ) -> Result<(SfmrReconstruction, RepeatObservationsReport), ReconstructionError> {
        progress.check_cancel()?;
        let set = &self.point_set;

        // Each group is the rows of one point in one image, in track order,
        // where there are two or more.
        let mut groups: Vec<Vec<usize>> = Vec::new();
        let mut changed = vec![false; set.points.len()];
        for (point, flag) in changed.iter_mut().enumerate() {
            let start = set.observation_offsets[point];
            let mut by_image: BTreeMap<u32, Vec<usize>> = BTreeMap::new();
            for (k, observation) in self.observations_for_point(point).iter().enumerate() {
                by_image
                    .entry(observation.image_index)
                    .or_default()
                    .push(start + k);
            }
            for rows in by_image.into_values().filter(|rows| rows.len() > 1) {
                *flag = true;
                groups.push(rows);
            }
        }
        if groups.is_empty() {
            return Ok((self.clone(), RepeatObservationsReport::default()));
        }
        progress.check_cancel()?;

        // The pixels of every observation of a changed point: the groups rank
        // on them, and the changed points are scored again on them.
        let mut images = vec![false; self.image_count()];
        for observation in &set.tracks {
            if changed[observation.point_index as usize] {
                images[observation.image_index as usize] = true;
            }
        }
        let pixels = self.observation_pixels(&images)?;
        progress.check_cancel()?;

        let intrinsics = self.image_set.all_image_intrinsics();
        let error_of = |row: usize| -> Option<f64> {
            let observation = set.tracks[row];
            let image = &self.image_set.images[observation.image_index as usize];
            let point = &set.points[observation.point_index as usize];
            let observed = [pixels[[row, 0]] as f64, pixels[[row, 1]] as f64];
            if !(observed[0].is_finite() && observed[1].is_finite()) {
                return None;
            }
            observation_reprojection_error(
                &image.quaternion_wxyz,
                &image.translation_xyz,
                &intrinsics[observation.image_index as usize],
                &point.position,
                point.is_at_infinity(),
                observed,
            )
            .filter(|e| e.is_finite())
        };

        let mut keep_row = vec![true; set.tracks.len()];
        let mut report = RepeatObservationsReport::default();
        for rows in &groups {
            // Strictly lower replaces, so a tie keeps the earlier row, and a
            // group where nothing projects keeps its first.
            let mut best = rows[0];
            let mut best_error = error_of(best);
            for &row in &rows[1..] {
                let error = error_of(row);
                let better = match (error, best_error) {
                    (Some(e), Some(b)) => e < b,
                    (Some(_), None) => true,
                    (None, _) => false,
                };
                if better {
                    best = row;
                    best_error = error;
                }
            }
            for &row in rows {
                keep_row[row] = row == best;
            }
            report.observations_dropped += rows.len() - 1;
        }
        report.points_changed = changed.iter().filter(|&&c| c).count();

        // Scored on the rows each changed point keeps, before the selection
        // renumbers them. No point is removed, so point indexes carry over.
        let rescored: Vec<(usize, f32)> = changed
            .iter()
            .enumerate()
            .filter(|&(_, &c)| c)
            .map(|(point, _)| {
                let start = set.observation_offsets[point];
                let observations = self
                    .observations_for_point(point)
                    .iter()
                    .enumerate()
                    .filter(|&(k, _)| keep_row[start + k])
                    .map(|(k, observation)| {
                        let row = start + k;
                        (
                            observation.image_index as usize,
                            [pixels[[row, 0]] as f64, pixels[[row, 1]] as f64],
                        )
                    })
                    .filter(|(_, xy)| xy[0].is_finite() && xy[1].is_finite());
                let error = rms_reprojection_error_px(
                    &self.image_set,
                    &intrinsics,
                    &set.points[point],
                    observations,
                );
                (point, error)
            })
            .collect();

        let (mut out, _) = self.select_rows(&keep_row, &vec![true; set.points.len()]);
        for (point, error) in rescored {
            out.point_set.points[point].rms_reprojection_error = error;
        }
        // The histograms count one entry per observation. The derivation also
        // writes the mean viewing normal of every point when the value carries
        // normals, which would replace normals this call does not mean to
        // touch, so it runs with them switched off.
        let has_normals = std::mem::replace(&mut out.point_set.has_normals, false);
        let derived = out.recompute_depth_statistics();
        out.point_set.has_normals = has_normals;
        derived.map_err(|e| ReconstructionError::Unsupported(e.to_string()))?;

        progress_info!(
            progress,
            "dropped {} repeat observations from {} points",
            report.observations_dropped,
            report.points_changed
        );
        Ok((out, report))
    }
}

#[cfg(test)]
mod tests;
