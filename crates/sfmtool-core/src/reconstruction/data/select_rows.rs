// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Taking observation rows and points out of a reconstruction, with every
//! per-point and per-observation column kept in step.

use std::sync::Arc;

use ndarray::Axis;

use sfmtool_sfmr_format::ContentHash;

use super::{ObservationSource, Point3D, PointSet, SfmrReconstruction, TrackObservation};

impl SfmrReconstruction {
    /// This value with the observation rows `keep_row` refuses and the points
    /// `keep_point` refuses taken out, and the surviving points renumbered,
    /// together with one flag per surviving point, in the output's order, that
    /// is true where the point lost a row.
    ///
    /// `keep_row` has one entry per row of `tracks` and `keep_point` one per
    /// point. A row of a refused point goes with its point whatever `keep_row`
    /// says.
    ///
    /// Every column but the tracks travels verbatim: a surviving point keeps
    /// its position, its frame, its bitmap, its colour, its normal and its
    /// constraint, and only its observation list is shorter. Every
    /// per-observation column (feature indexes, inline keypoints, confidences,
    /// readings) takes the same row selection, a reference observation moves
    /// with its row or becomes `-1` when its row goes, and the image set,
    /// depth statistics included, is shared unchanged. The stored RMS
    /// reprojection errors are carried as they are: a caller that removed rows
    /// scores the flagged points again.
    pub(crate) fn select_rows(
        &self,
        keep_row: &[bool],
        keep_point: &[bool],
    ) -> (SfmrReconstruction, Vec<bool>) {
        let set = &self.point_set;
        let keep_idx: Vec<usize> = (0..set.points.len()).filter(|&p| keep_point[p]).collect();
        let mut point_remap = vec![u32::MAX; set.points.len()];
        for (new, &old) in keep_idx.iter().enumerate() {
            point_remap[old] = new as u32;
        }
        let kept_obs: Vec<usize> = (0..set.tracks.len())
            .filter(|&row| keep_row[row] && keep_point[set.tracks[row].point_index as usize])
            .collect();

        let points: Vec<Point3D> = keep_idx.iter().map(|&p| set.points[p].clone()).collect();
        let mut observation_counts = vec![0u32; keep_idx.len()];
        let tracks: Vec<TrackObservation> = kept_obs
            .iter()
            .map(|&row| {
                let point = point_remap[set.tracks[row].point_index as usize];
                observation_counts[point as usize] += 1;
                TrackObservation {
                    image_index: set.tracks[row].image_index,
                    point_index: point,
                }
            })
            .collect();

        let observations = match &set.observations {
            ObservationSource::SiftFiles {
                feature_indexes,
                keypoints_xy,
                feature_tool_hashes,
                sift_content_hashes,
            } => ObservationSource::SiftFiles {
                feature_indexes: kept_obs.iter().map(|&row| feature_indexes[row]).collect(),
                keypoints_xy: keypoints_xy
                    .as_ref()
                    .map(|kp| kp.select(Axis(0), &kept_obs)),
                feature_tool_hashes: feature_tool_hashes.clone(),
                sift_content_hashes: sift_content_hashes.clone(),
            },
            ObservationSource::EmbeddedPatches {
                keypoints_xy,
                image_file_hashes,
            } => ObservationSource::EmbeddedPatches {
                keypoints_xy: keypoints_xy.select(Axis(0), &kept_obs),
                image_file_hashes: image_file_hashes.clone(),
            },
        };

        let mut point_set = PointSet {
            points,
            tracks,
            observation_counts,
            observations,
            patch_u_halfvec_xyz: set
                .patch_u_halfvec_xyz
                .as_ref()
                .map(|a| a.select(Axis(0), &keep_idx)),
            patch_v_halfvec_xyz: set
                .patch_v_halfvec_xyz
                .as_ref()
                .map(|a| a.select(Axis(0), &keep_idx)),
            patch_bitmaps_y_x_rgba: set
                .patch_bitmaps_y_x_rgba
                .as_ref()
                .map(|a| Arc::new(a.select(Axis(0), &keep_idx))),
            patch_bitmaps_for_display: set.patch_bitmaps_for_display,
            has_normals: set.has_normals,
            normal_confidence: set
                .normal_confidence
                .as_ref()
                .map(|c| keep_idx.iter().map(|&p| c[p]).collect()),
            // A constraint describes its own point and the images are
            // untouched, so the surviving rows travel verbatim.
            point_constraints: set.point_constraints.as_ref().map(|c| c.select(&keep_idx)),
            observation_confidence: set
                .observation_confidence
                .as_ref()
                .map(|c| kept_obs.iter().map(|&row| c[row]).collect()),
            observation_readings: set
                .observation_readings
                .as_ref()
                .map(|r| r.select(&kept_obs)),
            // A point whose reference observation is taken out keeps its
            // bitmap and loses its reference.
            reference_observations: set.select_reference_observations(&keep_idx, &kept_obs),
            display_only_references: set.select_display_only_references(&keep_idx),
            observation_offsets: Vec::new(),
            image_feature_to_point: Vec::new(),
            max_track_feature_index: Vec::new(),
            infinity_point_count: 0,
        };
        point_set.rebuild_derived_fields(self.image_count());

        let lost_observation: Vec<bool> = keep_idx
            .iter()
            .zip(&point_set.observation_counts)
            .map(|(&old, &kept)| kept < set.observation_counts[old])
            .collect();

        let mut metadata = self.metadata.clone();
        metadata.point_count = point_set.points.len() as u32;
        metadata.observation_count = point_set.tracks.len() as u32;
        metadata.infinity_point_count = point_set.infinity_point_count as u32;
        let selected = SfmrReconstruction {
            workspace_dir: self.workspace_dir.clone(),
            metadata,
            // The hashes on a value describe the file it came from, and this
            // value is not that file.
            content_hash: ContentHash::default(),
            // No image is touched, so the whole image set is the source's,
            // thumbnails shared rather than copied.
            image_set: self.image_set.clone(),
            point_set,
        };
        (selected, lost_observation)
    }
}
