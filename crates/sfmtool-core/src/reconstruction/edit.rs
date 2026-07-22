// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Editing operations on [`super::SfmrReconstruction`] that each return a new
//! reconstruction: SE(3) transform, image subsetting, and point-mask filtering.
//!
//! Split out from `reconstruction.rs` so the parent module keeps the data
//! types, accessors, and file I/O. These are inherent methods on
//! `SfmrReconstruction`, so callers are unaffected by the move.

use std::collections::HashMap;

use nalgebra::Point3;
use ndarray::Array4;

use sfmr_format::{DepthStatistics, FramesMetadata, ImageDepthStats, RigFrameData};

use super::data::{compute_observation_offsets, count_points_at_infinity};
use super::*;

/// Select rows `idx` (along the point axis) of an optional per-point patch
/// half-vector array, preserving `None`. Used to carry a patch frame through
/// the point reindexing that subset/filter perform.
fn select_patch_rows_f32(
    arr: &Option<ndarray::Array2<f32>>,
    idx: &[usize],
) -> Option<ndarray::Array2<f32>> {
    arr.as_ref().map(|a| a.select(ndarray::Axis(0), idx))
}

/// Like [`select_patch_rows_f32`] for the `(P, R, R, 4)` patch-bitmap array.
fn select_patch_rows_u8(
    arr: &Option<ndarray::Array4<u8>>,
    idx: &[usize],
) -> Option<ndarray::Array4<u8>> {
    arr.as_ref().map(|a| a.select(ndarray::Axis(0), idx))
}

impl SfmrReconstruction {
    /// Apply an SE(3) similarity transform to this reconstruction.
    ///
    /// Transforms all 3D point positions and camera poses. Returns a new
    /// reconstruction with the transformed data; `self` is not modified.
    pub fn apply_se3_transform(&self, transform: &crate::Se3Transform) -> Self {
        use crate::geometry::RotQuaternion;

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
                // A normal is a direction: only the rotation acts on it.
                let n = nalgebra::Vector3::new(
                    pt.normal.x as f64,
                    pt.normal.y as f64,
                    pt.normal.z as f64,
                );
                let rn = transform.rotation.rotate_vector(&n);
                let normal = nalgebra::Vector3::new(rn.x as f32, rn.y as f32, rn.z as f32);
                Point3D {
                    position,
                    normal,
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

        // A patch frame travels with its point under the similarity. The
        // in-plane half-vectors are world-space vectors: the rotation reorients
        // them and, for finite points, the scale resizes them (an infinity-point
        // patch is angular — tangent to the sphere of directions — so only the
        // rotation acts, matching the position handling above). Empty (no-patch)
        // rows stay zero. The `u × v` normal thus rotates in step with the
        // per-point `normal` rotated above.
        let transform_halfvec =
            |arr: &Option<ndarray::Array2<f32>>| -> Option<ndarray::Array2<f32>> {
                arr.as_ref().map(|a| {
                    let mut out = a.clone();
                    for (i, pt) in self.points.iter().enumerate() {
                        let v = nalgebra::Vector3::new(
                            a[[i, 0]] as f64,
                            a[[i, 1]] as f64,
                            a[[i, 2]] as f64,
                        );
                        let r = transform.rotation.rotate_vector(&v);
                        let scaled = if pt.is_at_infinity() {
                            r
                        } else {
                            r * transform.scale
                        };
                        out[[i, 0]] = scaled.x as f32;
                        out[[i, 1]] = scaled.y as f32;
                        out[[i, 2]] = scaled.z as f32;
                    }
                    out
                })
            };
        let new_patch_u = transform_halfvec(&self.patch_u_halfvec_xyz);
        let new_patch_v = transform_halfvec(&self.patch_v_halfvec_xyz);

        let infinity_point_count = count_points_at_infinity(&new_points);
        SfmrReconstruction {
            infinity_point_count,
            images: new_images,
            points: new_points,
            rig_frame_data: new_rig_frame_data,
            patch_u_halfvec_xyz: new_patch_u,
            patch_v_halfvec_xyz: new_patch_v,
            // Bitmaps live in the patch's own (s, t) frame, so a similarity of
            // the whole scene leaves their appearance unchanged.
            patch_bitmaps_y_x_rgba: self.patch_bitmaps_y_x_rgba.clone(),
            has_normals: self.has_normals,
            // A 3D similarity leaves the 2D image keypoints, feature indices, and
            // image identity untouched, so the observation source passes through
            // for both modes.
            observations: self.observations.clone(),
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
        // Works for both observation sources: the per-observation parallel
        // column (SiftFiles `feature_indexes` / EmbeddedPatches `keypoints_xy`
        // rows) is filtered in lockstep with the tracks below, and the per-image
        // hashes follow the image subset.
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
        // a simple single-pass filter preserves that grouping. The parallel
        // feature_indexes column is filtered in lockstep (point-id remapping
        // below preserves order, so it stays parallel to the final tracks).
        let mut new_tracks: Vec<TrackObservation> = Vec::with_capacity(self.tracks.len());
        let mut kept_obs: Vec<usize> = Vec::with_capacity(self.tracks.len());
        for (i, obs) in self.tracks.iter().enumerate() {
            if let Some(new_img_idx) = old_to_new[obs.image_index as usize] {
                new_tracks.push(TrackObservation {
                    image_index: new_img_idx,
                    point_index: obs.point_index,
                });
                kept_obs.push(i);
            }
        }

        // Points + observation counts. A patch frame is per-point, so it rides
        // along: subset keeps the rows for surviving points (geometry is
        // unchanged, so the frame vectors stay valid).
        let (
            new_points,
            new_observation_counts,
            new_tracks,
            new_patch_u,
            new_patch_v,
            new_patch_bitmaps,
        ) = if drop_orphaned_points {
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

            let keep_idx: Vec<usize> = keep_mask
                .iter()
                .enumerate()
                .filter(|(_, &k)| k)
                .map(|(i, _)| i)
                .collect();
            (
                kept_points,
                kept_counts,
                remapped_tracks,
                select_patch_rows_f32(&self.patch_u_halfvec_xyz, &keep_idx),
                select_patch_rows_f32(&self.patch_v_halfvec_xyz, &keep_idx),
                select_patch_rows_u8(&self.patch_bitmaps_y_x_rgba, &keep_idx),
            )
        } else {
            // Keep all points; recompute per-point counts from the filtered tracks.
            let mut per_point_count = vec![0u32; self.points.len()];
            for obs in &new_tracks {
                per_point_count[obs.point_index as usize] += 1;
            }
            (
                self.points.clone(),
                per_point_count,
                new_tracks,
                self.patch_u_halfvec_xyz.clone(),
                self.patch_v_halfvec_xyz.clone(),
                self.patch_bitmaps_y_x_rgba.clone(),
            )
        };

        let new_observation_offsets = compute_observation_offsets(&new_observation_counts);

        // Rebuild the observation source: the per-observation parallel column is
        // subset by the kept-observation indices, the per-image hashes by the
        // image subset. Matches the input variant.
        let new_observations = match &self.observations {
            ObservationSource::SiftFiles {
                feature_indexes,
                feature_tool_hashes,
                sift_content_hashes,
            } => ObservationSource::SiftFiles {
                feature_indexes: kept_obs.iter().map(|&i| feature_indexes[i]).collect(),
                feature_tool_hashes: image_indices
                    .iter()
                    .map(|&i| feature_tool_hashes[i as usize])
                    .collect(),
                sift_content_hashes: image_indices
                    .iter()
                    .map(|&i| sift_content_hashes[i as usize])
                    .collect(),
            },
            ObservationSource::EmbeddedPatches {
                keypoints_xy,
                image_file_hashes,
            } => {
                let mut new_keypoints = ndarray::Array2::<f32>::zeros((kept_obs.len(), 2));
                for (new_i, &old_i) in kept_obs.iter().enumerate() {
                    new_keypoints[[new_i, 0]] = keypoints_xy[[old_i, 0]];
                    new_keypoints[[new_i, 1]] = keypoints_xy[[old_i, 1]];
                }
                ObservationSource::EmbeddedPatches {
                    keypoints_xy: new_keypoints,
                    image_file_hashes: image_indices
                        .iter()
                        .map(|&i| image_file_hashes[i as usize])
                        .collect(),
                }
            }
        };

        // The feature→point maps are meaningful only for SiftFiles; an
        // embedded_patches reconstruction leaves them empty.
        let mut new_image_feature_to_point = vec![HashMap::new(); new_image_count];
        let mut new_max_track_feature_index = vec![0u32; new_image_count];
        if let ObservationSource::SiftFiles {
            feature_indexes, ..
        } = &new_observations
        {
            for (obs, &feat) in new_tracks.iter().zip(feature_indexes) {
                let img = obs.image_index as usize;
                new_image_feature_to_point[img].insert(feat, obs.point_index);
                new_max_track_feature_index[img] = new_max_track_feature_index[img].max(feat);
            }
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
            patch_u_halfvec_xyz: new_patch_u,
            patch_v_halfvec_xyz: new_patch_v,
            patch_bitmaps_y_x_rgba: new_patch_bitmaps,
            has_normals: self.has_normals,
            observations: new_observations,
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

        // The patch frame is per-point: keep the rows for surviving points
        // (geometry is unchanged, so the frame vectors stay valid).
        let keep_idx: Vec<usize> = mask
            .iter()
            .enumerate()
            .filter(|(_, &k)| k)
            .map(|(i, _)| i)
            .collect();
        let new_patch_u = select_patch_rows_f32(&self.patch_u_halfvec_xyz, &keep_idx);
        let new_patch_v = select_patch_rows_f32(&self.patch_v_halfvec_xyz, &keep_idx);
        let new_patch_bitmaps = select_patch_rows_u8(&self.patch_bitmaps_y_x_rgba, &keep_idx);

        let new_observation_counts: Vec<u32> = self
            .observation_counts
            .iter()
            .zip(mask.iter())
            .filter(|(_, &keep)| keep)
            .map(|(&count, _)| count)
            .collect();

        // Remap surviving point ids to be contiguous, then filter observations
        // (preserving order) and the parallel observation-source column. Works
        // for both modes — keypoints are filtered in lockstep with the tracks.
        let mut point_remap = vec![u32::MAX; self.points.len()];
        let mut next_id = 0u32;
        for (old, &keep) in mask.iter().enumerate() {
            if keep {
                point_remap[old] = next_id;
                next_id += 1;
            }
        }
        let kept: Vec<usize> = (0..self.tracks.len())
            .filter(|&i| mask[self.tracks[i].point_index as usize])
            .collect();
        let new_tracks: Vec<TrackObservation> = kept
            .iter()
            .map(|&i| TrackObservation {
                image_index: self.tracks[i].image_index,
                point_index: point_remap[self.tracks[i].point_index as usize],
            })
            .collect();

        // Images are unchanged, so per-image hashes pass through; the
        // per-observation column is filtered by `kept`.
        let new_observations = match &self.observations {
            ObservationSource::SiftFiles {
                feature_indexes,
                feature_tool_hashes,
                sift_content_hashes,
            } => ObservationSource::SiftFiles {
                feature_indexes: kept.iter().map(|&i| feature_indexes[i]).collect(),
                feature_tool_hashes: feature_tool_hashes.clone(),
                sift_content_hashes: sift_content_hashes.clone(),
            },
            ObservationSource::EmbeddedPatches {
                keypoints_xy,
                image_file_hashes,
            } => ObservationSource::EmbeddedPatches {
                keypoints_xy: keypoints_xy.select(ndarray::Axis(0), &kept),
                image_file_hashes: image_file_hashes.clone(),
            },
        };

        let new_observation_offsets = compute_observation_offsets(&new_observation_counts);

        // Rebuild per-image feature→point mapping (sift_files only).
        let image_count = self.images.len();
        let mut new_image_feature_to_point = vec![HashMap::new(); image_count];
        let mut new_max_track_feature_index = vec![0u32; image_count];
        if let ObservationSource::SiftFiles {
            feature_indexes, ..
        } = &new_observations
        {
            for (obs, &feat) in new_tracks.iter().zip(feature_indexes) {
                let img = obs.image_index as usize;
                new_image_feature_to_point[img].insert(feat, obs.point_index);
                new_max_track_feature_index[img] = new_max_track_feature_index[img].max(feat);
            }
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
            patch_u_halfvec_xyz: new_patch_u,
            patch_v_halfvec_xyz: new_patch_v,
            patch_bitmaps_y_x_rgba: new_patch_bitmaps,
            has_normals: self.has_normals,
            observations: new_observations,
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

#[cfg(test)]
mod patch_frame_tests {
    use super::*;
    use crate::geometry::RotQuaternion;
    use crate::Se3Transform;
    use nalgebra::{UnitQuaternion, Vector3 as V3};
    use ndarray::{Array2, Array4};

    /// A demo reconstruction with a per-point patch frame attached: `u` along
    /// +x and `v` along +y (so `u × v` is +z), plus distinct-per-cell bitmaps.
    fn demo_with_patches() -> SfmrReconstruction {
        let mut recon = SfmrReconstruction::demo(4);
        let p = recon.points.len();
        let mut u = Array2::<f32>::zeros((p, 3));
        let mut v = Array2::<f32>::zeros((p, 3));
        for i in 0..p {
            u[[i, 0]] = 0.1 * (i + 1) as f32;
            v[[i, 1]] = 0.2 * (i + 1) as f32;
        }
        let bitmaps = Array4::<u8>::from_shape_fn((p, 2, 2, 4), |(i, y, x, c)| {
            ((i * 13 + y * 5 + x * 3 + c) % 256) as u8
        });
        recon.patch_u_halfvec_xyz = Some(u);
        recon.patch_v_halfvec_xyz = Some(v);
        recon.patch_bitmaps_y_x_rgba = Some(bitmaps);
        recon
    }

    fn approx(a: f64, b: f64) {
        assert!((a - b).abs() < 1e-5, "{a} != {b}");
    }

    #[test]
    fn se3_transform_rotates_and_scales_patch_frame_and_normals() {
        let recon = demo_with_patches();
        let u0 = recon.patch_u_halfvec_xyz.clone().unwrap();
        let v0 = recon.patch_v_halfvec_xyz.clone().unwrap();
        let bitmaps0 = recon.patch_bitmaps_y_x_rgba.clone().unwrap();
        let n0: Vec<_> = recon.points.iter().map(|p| p.normal).collect();

        // 90° about +z, uniform scale 2, arbitrary translation.
        let rot = RotQuaternion::from_nalgebra(UnitQuaternion::from_axis_angle(
            &V3::z_axis(),
            std::f64::consts::FRAC_PI_2,
        ));
        let t = Se3Transform::new(rot.clone(), V3::new(1.0, 2.0, 3.0), 2.0);
        let out = recon.apply_se3_transform(&t);

        // Bitmaps are pose-invariant: carried byte-for-byte.
        assert_eq!(out.patch_bitmaps_y_x_rgba.as_ref().unwrap(), &bitmaps0);

        let u1 = out.patch_u_halfvec_xyz.as_ref().unwrap();
        let v1 = out.patch_v_halfvec_xyz.as_ref().unwrap();
        for i in 0..recon.points.len() {
            // Half-vectors: rotated by R and scaled by s.
            for (a0, a1) in [(&u0, u1), (&v0, v1)] {
                let src = V3::new(a0[[i, 0]] as f64, a0[[i, 1]] as f64, a0[[i, 2]] as f64);
                let want = rot.rotate_vector(&src) * t.scale;
                approx(a1[[i, 0]] as f64, want.x);
                approx(a1[[i, 1]] as f64, want.y);
                approx(a1[[i, 2]] as f64, want.z);
            }
            // Normal: a direction, rotated by R (no scale, stays unit).
            let nn = V3::new(n0[i].x as f64, n0[i].y as f64, n0[i].z as f64);
            let want_n = rot.rotate_vector(&nn);
            approx(out.points[i].normal.x as f64, want_n.x);
            approx(out.points[i].normal.y as f64, want_n.y);
            approx(out.points[i].normal.z as f64, want_n.z);
        }

        // The frame stays rigid: normalize(u × v) just rotates by R. Check pt 0,
        // whose pre-transform u × v is +z.
        let u1v = V3::new(u1[[0, 0]] as f64, u1[[0, 1]] as f64, u1[[0, 2]] as f64);
        let v1v = V3::new(v1[[0, 0]] as f64, v1[[0, 1]] as f64, v1[[0, 2]] as f64);
        let n_patch = u1v.cross(&v1v).normalize();
        let want = rot.rotate_vector(&V3::z());
        approx(n_patch.x, want.x);
        approx(n_patch.y, want.y);
        approx(n_patch.z, want.z);
    }

    #[test]
    fn filter_keeps_patch_rows_for_surviving_points() {
        let recon = demo_with_patches();
        let u0 = recon.patch_u_halfvec_xyz.clone().unwrap();
        let mask = vec![true, false, true, false];
        let out = recon.filter_points_by_mask(&mask);

        assert_eq!(out.point_count(), 2);
        let u1 = out.patch_u_halfvec_xyz.as_ref().unwrap();
        assert_eq!(u1.shape(), &[2, 3]);
        // Kept rows are the source rows 0 and 2, unchanged.
        approx(u1[[0, 0]] as f64, u0[[0, 0]] as f64);
        approx(u1[[1, 0]] as f64, u0[[2, 0]] as f64);
        assert_eq!(out.patch_bitmaps_y_x_rgba.as_ref().unwrap().shape()[0], 2);
    }

    #[test]
    fn subset_keeping_all_images_carries_the_patch_frame() {
        let recon = demo_with_patches();
        let u0 = recon.patch_u_halfvec_xyz.clone().unwrap();
        let all: Vec<u32> = (0..recon.images.len() as u32).collect();
        let out = recon.subset_by_image_indices(&all, true).unwrap();

        assert_eq!(out.point_count(), recon.point_count());
        assert_eq!(out.patch_u_halfvec_xyz.as_ref().unwrap(), &u0);
        assert_eq!(
            out.patch_bitmaps_y_x_rgba.as_ref().unwrap(),
            recon.patch_bitmaps_y_x_rgba.as_ref().unwrap()
        );
    }
}
