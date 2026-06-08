// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Editing operations on [`super::SfmrReconstruction`] that each return a new
//! reconstruction: SE(3) transform, image subsetting, and point-mask filtering.
//!
//! Split out from `reconstruction.rs` so the parent module keeps the data
//! types, accessors, and file I/O. These are inherent methods on
//! `SfmrReconstruction`, so callers are unaffected by the move.

use super::*;

impl SfmrReconstruction {
    /// Apply an SE(3) similarity transform to this reconstruction.
    ///
    /// Transforms all 3D point positions and camera poses. Returns a new
    /// reconstruction with the transformed data; `self` is not modified.
    pub fn apply_se3_transform(&self, transform: &crate::Se3Transform) -> Self {
        use crate::rot_quaternion::RotQuaternion;

        // Transform 3D points. A finite point gets the full similarity; a
        // point at infinity is a direction, so only the rotation acts on it
        // (translation and scale do not move a point at infinity) and the
        // result is renormalised to keep the stored direction unit-length.
        let point_positions: Vec<Point3<f64>> = self.points.iter().map(|pt| pt.position).collect();
        let new_positions = transform.apply_to_points(&point_positions);

        let new_points: Vec<Point3D> = self
            .points
            .iter()
            .zip(new_positions.iter())
            .map(|(pt, &new_pos)| {
                let position = if pt.is_at_infinity() {
                    let rotated = transform.rotation.rotate_vector(&pt.position.coords);
                    let norm = rotated.norm();
                    Point3::from(if norm > 0.0 { rotated / norm } else { rotated })
                } else {
                    new_pos
                };
                Point3D {
                    position,
                    ..pt.clone()
                }
            })
            .collect();

        // Transform camera poses
        let new_images: Vec<SfmrImage> = self
            .images
            .iter()
            .map(|im| {
                let cam_rot = RotQuaternion::from_nalgebra(im.quaternion_wxyz);
                let (new_rot, new_t) =
                    transform.apply_to_camera_pose(&cam_rot, &im.translation_xyz);
                SfmrImage {
                    quaternion_wxyz: *new_rot.as_nalgebra(),
                    translation_xyz: new_t,
                    ..im.clone()
                }
            })
            .collect();

        // Scale rig sensor translations if needed
        let new_rig_frame_data = if let Some(ref rf) = self.rig_frame_data {
            let scale = transform.scale;
            if (scale - 1.0).abs() > f64::EPSILON {
                let mut rf = rf.clone();
                rf.sensor_translations_xyz.mapv_inplace(|v| v * scale);
                Some(rf)
            } else {
                Some(rf.clone())
            }
        } else {
            None
        };

        let infinity_point_count = count_points_at_infinity(&new_points);
        SfmrReconstruction {
            infinity_point_count,
            images: new_images,
            points: new_points,
            rig_frame_data: new_rig_frame_data,
            // All other fields are unchanged
            workspace_dir: self.workspace_dir.clone(),
            metadata: self.metadata.clone(),
            content_hash: self.content_hash.clone(),
            cameras: self.cameras.clone(),
            tracks: self.tracks.clone(),
            observation_counts: self.observation_counts.clone(),
            observation_offsets: self.observation_offsets.clone(),
            thumbnails_y_x_rgb: self.thumbnails_y_x_rgb.clone(),
            depth_statistics: self.depth_statistics.clone(),
            depth_histogram_counts: self.depth_histogram_counts.clone(),
            image_feature_to_point: self.image_feature_to_point.clone(),
            max_track_feature_index: self.max_track_feature_index.clone(),
        }
    }

    /// Return a new reconstruction that retains only the images at
    /// `image_indices` (0-based, in the order given). Observations whose
    /// image is not kept are dropped. Frames with no kept image are dropped;
    /// rig and sensor definitions are preserved.
    ///
    /// If `drop_orphaned_points` is true, 3D points with zero remaining
    /// observations are removed and point IDs are remapped to be contiguous.
    /// Otherwise, all 3D points are kept with their original IDs (some may
    /// have `observation_count == 0`).
    ///
    /// # Errors
    /// Returns an error if any index is out of bounds or if `image_indices`
    /// contains duplicates.
    pub fn subset_by_image_indices(
        &self,
        image_indices: &[u32],
        drop_orphaned_points: bool,
    ) -> Result<Self, String> {
        let old_image_count = self.images.len();
        let new_image_count = image_indices.len();

        // Validate: in bounds, no duplicates. old_to_new is None for removed
        // images, Some(new_idx) for kept images (new_idx matches the position
        // in `image_indices`).
        let mut old_to_new: Vec<Option<u32>> = vec![None; old_image_count];
        for (new_idx, &old_idx) in image_indices.iter().enumerate() {
            let old_idx_usize = old_idx as usize;
            if old_idx_usize >= old_image_count {
                return Err(format!(
                    "image index {} out of bounds (image_count={})",
                    old_idx, old_image_count
                ));
            }
            if old_to_new[old_idx_usize].is_some() {
                return Err(format!("duplicate image index {} in selection", old_idx));
            }
            old_to_new[old_idx_usize] = Some(new_idx as u32);
        }

        // Build new image-indexed data in the order given.
        let new_images: Vec<SfmrImage> = image_indices
            .iter()
            .map(|&i| self.images[i as usize].clone())
            .collect();

        let new_thumbnails = {
            let mut out = Array4::<u8>::zeros((new_image_count, 128, 128, 3));
            for (new_idx, &old_idx) in image_indices.iter().enumerate() {
                let src = self
                    .thumbnails_y_x_rgb
                    .slice(ndarray::s![old_idx as usize, .., .., ..]);
                out.slice_mut(ndarray::s![new_idx, .., .., ..]).assign(&src);
            }
            out
        };

        let new_depth_stats_images: Vec<ImageDepthStats> = image_indices
            .iter()
            .map(|&i| self.depth_statistics.images[i as usize].clone())
            .collect();
        let new_depth_statistics = DepthStatistics {
            num_histogram_buckets: self.depth_statistics.num_histogram_buckets,
            images: new_depth_stats_images,
        };

        let new_depth_histogram_counts: Vec<Vec<u32>> = image_indices
            .iter()
            .map(|&i| self.depth_histogram_counts[i as usize].clone())
            .collect();

        // Filter and remap tracks. Input tracks are grouped by point_index, so
        // a simple single-pass filter preserves that grouping.
        let mut new_tracks: Vec<TrackObservation> = Vec::with_capacity(self.tracks.len());
        for obs in &self.tracks {
            if let Some(new_img_idx) = old_to_new[obs.image_index as usize] {
                new_tracks.push(TrackObservation {
                    image_index: new_img_idx,
                    feature_index: obs.feature_index,
                    point_index: obs.point_index,
                });
            }
        }

        // Points + observation counts.
        let (new_points, new_observation_counts, new_tracks) = if drop_orphaned_points {
            // Count surviving observations per point and build a keep mask.
            let mut per_point_count = vec![0u32; self.points.len()];
            for obs in &new_tracks {
                per_point_count[obs.point_index as usize] += 1;
            }
            let keep_mask: Vec<bool> = per_point_count.iter().map(|&c| c > 0).collect();

            // Remap point ids to be contiguous over kept points.
            let mut point_remap = vec![u32::MAX; self.points.len()];
            let mut next_id = 0u32;
            for (old_id, &keep) in keep_mask.iter().enumerate() {
                if keep {
                    point_remap[old_id] = next_id;
                    next_id += 1;
                }
            }
            let remapped_tracks: Vec<TrackObservation> = new_tracks
                .into_iter()
                .map(|obs| TrackObservation {
                    image_index: obs.image_index,
                    feature_index: obs.feature_index,
                    point_index: point_remap[obs.point_index as usize],
                })
                .collect();

            let kept_points: Vec<Point3D> = self
                .points
                .iter()
                .zip(keep_mask.iter())
                .filter(|(_, &k)| k)
                .map(|(p, _)| p.clone())
                .collect();
            let kept_counts: Vec<u32> = per_point_count
                .iter()
                .zip(keep_mask.iter())
                .filter(|(_, &k)| k)
                .map(|(&c, _)| c)
                .collect();

            (kept_points, kept_counts, remapped_tracks)
        } else {
            // Keep all points; recompute per-point counts from the filtered tracks.
            let mut per_point_count = vec![0u32; self.points.len()];
            for obs in &new_tracks {
                per_point_count[obs.point_index as usize] += 1;
            }
            (self.points.clone(), per_point_count, new_tracks)
        };

        let new_observation_offsets = compute_observation_offsets(&new_observation_counts);

        // Rebuild per-image feature→point mapping and max feature indexes.
        let mut new_image_feature_to_point = vec![HashMap::new(); new_image_count];
        let mut new_max_track_feature_index = vec![0u32; new_image_count];
        for obs in &new_tracks {
            let img = obs.image_index as usize;
            new_image_feature_to_point[img].insert(obs.feature_index, obs.point_index);
            new_max_track_feature_index[img] =
                new_max_track_feature_index[img].max(obs.feature_index);
        }

        // Filter rig/frame data.
        let new_rig_frame_data = self
            .rig_frame_data
            .as_ref()
            .map(|rf| subset_rig_frame_data(rf, image_indices));

        let infinity_point_count = count_points_at_infinity(&new_points);
        Ok(SfmrReconstruction {
            infinity_point_count,
            workspace_dir: self.workspace_dir.clone(),
            metadata: self.metadata.clone(),
            content_hash: self.content_hash.clone(),
            cameras: self.cameras.clone(),
            images: new_images,
            points: new_points,
            tracks: new_tracks,
            observation_counts: new_observation_counts,
            observation_offsets: new_observation_offsets,
            thumbnails_y_x_rgb: new_thumbnails,
            depth_statistics: new_depth_statistics,
            depth_histogram_counts: new_depth_histogram_counts,
            rig_frame_data: new_rig_frame_data,
            image_feature_to_point: new_image_feature_to_point,
            max_track_feature_index: new_max_track_feature_index,
        })
    }

    /// Filter 3D points by a boolean mask, returning a new reconstruction.
    ///
    /// Points where `mask[i]` is `true` are kept; others are removed.
    /// Tracks are filtered and remapped to contiguous point IDs.
    /// Image data (poses, cameras, metadata, depth stats) is copied unchanged.
    ///
    /// # Panics
    /// Panics if `mask.len() != self.points.len()`.
    pub fn filter_points_by_mask(&self, mask: &[bool]) -> Self {
        assert_eq!(
            mask.len(),
            self.points.len(),
            "mask length ({}) must match point count ({})",
            mask.len(),
            self.points.len()
        );

        // Filter points and observation counts
        let new_points: Vec<Point3D> = self
            .points
            .iter()
            .zip(mask.iter())
            .filter(|(_, &keep)| keep)
            .map(|(pt, _)| pt.clone())
            .collect();

        let new_observation_counts: Vec<u32> = self
            .observation_counts
            .iter()
            .zip(mask.iter())
            .filter(|(_, &keep)| keep)
            .map(|(&count, _)| count)
            .collect();

        // Filter and remap tracks using the existing filter function
        let track_image_indexes: Vec<u32> = self.tracks.iter().map(|t| t.image_index).collect();
        let track_feature_indexes: Vec<u32> = self.tracks.iter().map(|t| t.feature_index).collect();
        let track_point_ids: Vec<u32> = self.tracks.iter().map(|t| t.point_index).collect();

        let filtered = crate::filter::filter_tracks_by_point_mask(
            mask,
            &track_image_indexes,
            &track_feature_indexes,
            &track_point_ids,
        );

        let new_tracks: Vec<TrackObservation> = (0..filtered.track_image_indexes.len())
            .map(|i| TrackObservation {
                image_index: filtered.track_image_indexes[i],
                feature_index: filtered.track_feature_indexes[i],
                point_index: filtered.track_point_ids[i],
            })
            .collect();

        let new_observation_offsets = compute_observation_offsets(&new_observation_counts);

        // Rebuild per-image feature→point mapping
        let image_count = self.images.len();
        let mut new_image_feature_to_point = vec![HashMap::new(); image_count];
        let mut new_max_track_feature_index = vec![0u32; image_count];
        for obs in &new_tracks {
            let img = obs.image_index as usize;
            new_image_feature_to_point[img].insert(obs.feature_index, obs.point_index);
            new_max_track_feature_index[img] =
                new_max_track_feature_index[img].max(obs.feature_index);
        }

        let infinity_point_count = count_points_at_infinity(&new_points);
        SfmrReconstruction {
            infinity_point_count,
            workspace_dir: self.workspace_dir.clone(),
            metadata: self.metadata.clone(),
            content_hash: self.content_hash.clone(),
            cameras: self.cameras.clone(),
            images: self.images.clone(),
            points: new_points,
            tracks: new_tracks,
            observation_counts: new_observation_counts,
            observation_offsets: new_observation_offsets,
            thumbnails_y_x_rgb: self.thumbnails_y_x_rgb.clone(),
            depth_statistics: self.depth_statistics.clone(),
            depth_histogram_counts: self.depth_histogram_counts.clone(),
            rig_frame_data: self.rig_frame_data.clone(),
            image_feature_to_point: new_image_feature_to_point,
            max_track_feature_index: new_max_track_feature_index,
        }
    }
}

/// Build a new `RigFrameData` restricted to the given image subset.
///
/// `image_indices` gives the old image indices in their new order.
/// `old_to_new` maps old image index → new image index (None if removed).
///
/// Frames with no remaining images are dropped; surviving frame indices are
/// remapped to be contiguous. Rig and sensor definitions are preserved
/// unchanged.
fn subset_rig_frame_data(rf: &RigFrameData, image_indices: &[u32]) -> RigFrameData {
    use ndarray::Array1;

    let new_image_count = image_indices.len();

    // Per-image arrays: reindex using image_indices order.
    let mut new_image_sensor_indexes = Array1::<u32>::zeros(new_image_count);
    let mut new_image_frame_indexes = Array1::<u32>::zeros(new_image_count);
    for (new_idx, &old_idx) in image_indices.iter().enumerate() {
        new_image_sensor_indexes[new_idx] = rf.image_sensor_indexes[old_idx as usize];
        new_image_frame_indexes[new_idx] = rf.image_frame_indexes[old_idx as usize];
    }

    // Which frames still have an image? Keep their original order.
    let old_frame_count = rf.frames_metadata.frame_count as usize;
    let mut frame_kept = vec![false; old_frame_count];
    for &f in new_image_frame_indexes.iter() {
        frame_kept[f as usize] = true;
    }

    let new_frame_count = frame_kept.iter().filter(|&&k| k).count();

    if new_frame_count == old_frame_count {
        // All frames kept; leave rig_indexes and frame indexes alone.
        return RigFrameData {
            rigs_metadata: rf.rigs_metadata.clone(),
            sensor_camera_indexes: rf.sensor_camera_indexes.clone(),
            sensor_quaternions_wxyz: rf.sensor_quaternions_wxyz.clone(),
            sensor_translations_xyz: rf.sensor_translations_xyz.clone(),
            frames_metadata: FramesMetadata {
                frame_count: new_frame_count as u32,
            },
            rig_indexes: rf.rig_indexes.clone(),
            image_sensor_indexes: new_image_sensor_indexes,
            image_frame_indexes: new_image_frame_indexes,
        };
    }

    // Build old_frame → new_frame mapping and filter rig_indexes.
    let mut frame_remap = vec![u32::MAX; old_frame_count];
    let mut new_rig_indexes_vec = Vec::with_capacity(new_frame_count);
    let mut next_frame_idx = 0u32;
    for (old_frame_idx, &keep) in frame_kept.iter().enumerate() {
        if keep {
            frame_remap[old_frame_idx] = next_frame_idx;
            new_rig_indexes_vec.push(rf.rig_indexes[old_frame_idx]);
            next_frame_idx += 1;
        }
    }

    // Remap image_frame_indexes to the new contiguous frame space.
    for v in new_image_frame_indexes.iter_mut() {
        *v = frame_remap[*v as usize];
    }

    RigFrameData {
        rigs_metadata: rf.rigs_metadata.clone(),
        sensor_camera_indexes: rf.sensor_camera_indexes.clone(),
        sensor_quaternions_wxyz: rf.sensor_quaternions_wxyz.clone(),
        sensor_translations_xyz: rf.sensor_translations_xyz.clone(),
        frames_metadata: FramesMetadata {
            frame_count: new_frame_count as u32,
        },
        rig_indexes: Array1::from_vec(new_rig_indexes_vec),
        image_sensor_indexes: new_image_sensor_indexes,
        image_frame_indexes: new_image_frame_indexes,
    }
}
