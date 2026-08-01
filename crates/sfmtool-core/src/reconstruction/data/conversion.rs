// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The `.sfmr` file boundary for [`SfmrReconstruction`].
//!
//! Holds the round trip against [`sfmr_format::SfmrData`], the raw columnar I/O
//! representation: [`SfmrReconstruction::from_sfmr_data`] /
//! [`SfmrReconstruction::to_sfmr_data`] and the [`load`](SfmrReconstruction::load)
//! / [`save`](SfmrReconstruction::save) wrappers around them. Split out of
//! [`super`] so the data model and its serialization stay separable; the type
//! definitions and accessors live there.

use std::collections::HashMap;
use std::path::Path;

use nalgebra::{Point3, UnitQuaternion, Vector3};

use sfmr_format::{
    resolve_workspace_dir, SfmrCamera, SfmrData, SfmrError, FEATURE_SOURCE_EMBEDDED_PATCHES,
};

use crate::camera::CameraIntrinsics;

use super::{
    compute_observation_offsets, count_points_at_infinity, ObservationSource, Point3D, SfmrImage,
    SfmrReconstruction, TrackObservation,
};

/// Unit quaternion from raw WXYZ components, keeping the caller's bits when
/// they are already unit.
///
/// [`UnitQuaternion::new_normalize`] divides by the computed norm even when
/// the input is a unit quaternion, and because `‖q‖` itself rounds to
/// `1 ± ε`, the division can move every component by one ULP. A
/// reconstruction that merely round-trips its poses (load → save, or
/// `clone_with_changes` fed the accessors' own arrays) would then not compare
/// bit-equal to its source. So: when every component of the normalized result
/// is within `4 ε` of the input — i.e. the input was already unit to working
/// precision — the input bits are kept verbatim; otherwise the normalized
/// value is returned exactly as before. Non-finite and zero inputs take the
/// normalize path unchanged (its NaNs fail the closeness test).
pub fn unit_quaternion_preserving(qw: f64, qx: f64, qy: f64, qz: f64) -> UnitQuaternion<f64> {
    let q = nalgebra::Quaternion::new(qw, qx, qy, qz);
    let normalized = UnitQuaternion::new_normalize(q);
    let tol = 4.0 * f64::EPSILON;
    let close = normalized
        .as_ref()
        .coords
        .iter()
        .zip(q.coords.iter())
        .all(|(a, b)| (a - b).abs() <= tol);
    if close {
        UnitQuaternion::new_unchecked(q)
    } else {
        normalized
    }
}

impl SfmrReconstruction {
    /// Load a reconstruction from a `.sfmr` file.
    ///
    /// Resolves the workspace directory using the strategy from the spec:
    /// 1. Try `workspace.relative_path` from the `.sfmr` file's directory
    /// 2. Fall back to `workspace.absolute_path`
    /// 3. Fall back to searching upward from the `.sfmr` file for `.sfm-workspace.json`
    ///
    /// Version ≤ 4 files store COLMAP-convention data and are upgraded to the
    /// canonical convention here (`.sfmr` version 5, design decision D1):
    /// `S` on camera and rig sensor poses, `W` on world points (including
    /// `w = 0` infinity directions), normals, and patch half-vectors — see
    /// [`crate::geometry::convention::sfmr_data_colmap_to_canonical`]. The
    /// conversion lives in this crate rather than `sfmr-format` because the
    /// convention math is `sfmtool-core`'s `geometry::convention`, which the
    /// lower-level format crate cannot depend on. Content hashes cover the
    /// stored bytes ([`sfmr_format::verify_sfmr`] re-reads the file), so
    /// integrity checks are unaffected; a subsequent [`save`](Self::save)
    /// writes a new version-5 file with new hashes.
    pub fn load(path: &Path) -> Result<Self, SfmrError> {
        let mut data = sfmr_format::read_sfmr(path)?;
        // read_sfmr resolves workspace best-effort; here we require it
        let workspace_dir = match data.workspace_dir {
            Some(ref dir) => dir.clone(),
            None => resolve_workspace_dir(path, &data.metadata)?,
        };
        if data.metadata.version < sfmr_format::SFMR_FORMAT_VERSION {
            crate::geometry::convention::sfmr_data_colmap_to_canonical(&mut data);
            data.metadata.version = sfmr_format::SFMR_FORMAT_VERSION;
        }
        let mut recon = Self::from_sfmr_data(data)?;
        recon.workspace_dir = workspace_dir;
        Ok(recon)
    }

    /// Save this reconstruction to a `.sfmr` file.
    ///
    /// The write preserves the in-memory `normal` of every point that
    /// has one, recomputing only the missing (zero) normals from geometry — so
    /// normals a consumer has set (e.g. `sfm xform --refine-normals`) survive the
    /// round trip. Depth statistics and histograms are still recomputed.
    pub fn save(&self, path: &Path) -> Result<(), SfmrError> {
        let mut data = self.to_sfmr_data();
        sfmr_format::write_sfmr(path, &mut data)
    }

    /// Convert from the raw columnar I/O representation.
    pub fn from_sfmr_data(data: SfmrData) -> Result<Self, SfmrError> {
        // Both observation sources load; the mode picks which columns the
        // `ObservationSource` enum carries (built at the end).
        let is_embedded = data.metadata.feature_source == FEATURE_SOURCE_EMBEDDED_PATCHES;

        let image_count = data.metadata.image_count as usize;
        let point_count = data.metadata.point_count as usize;
        let observation_count = data.metadata.observation_count as usize;
        let num_buckets = data.depth_statistics.num_histogram_buckets as usize;

        // Convert images
        let mut images = Vec::with_capacity(image_count);
        for i in 0..image_count {
            let qw = data.quaternions_wxyz[[i, 0]];
            let qx = data.quaternions_wxyz[[i, 1]];
            let qy = data.quaternions_wxyz[[i, 2]];
            let qz = data.quaternions_wxyz[[i, 3]];
            let quaternion = unit_quaternion_preserving(qw, qx, qy, qz);

            let tx = data.translations_xyz[[i, 0]];
            let ty = data.translations_xyz[[i, 1]];
            let tz = data.translations_xyz[[i, 2]];

            images.push(SfmrImage {
                name: data.image_names[i].clone(),
                camera_index: data.camera_indexes[i],
                quaternion_wxyz: quaternion,
                translation_xyz: Vector3::new(tx, ty, tz),
            });
        }

        // Convert points. On-disk positions are homogeneous (x, y, z, w);
        // normalise into the ergonomic form — a finite point stores its
        // Euclidean position with w = 1, a point at infinity stores a
        // unit-length direction with w = 0.
        let has_normals = data.normals_xyz.is_some();
        let mut points = Vec::with_capacity(point_count);
        for i in 0..point_count {
            let x = data.positions_xyzw[[i, 0]];
            let y = data.positions_xyzw[[i, 1]];
            let z = data.positions_xyzw[[i, 2]];
            let w = data.positions_xyzw[[i, 3]];
            let (position, w) = if w != 0.0 {
                (Point3::new(x / w, y / w, z / w), 1.0)
            } else {
                let dir = Vector3::new(x, y, z);
                let norm = dir.norm();
                let unit = if norm > 0.0 { dir / norm } else { dir };
                (Point3::from(unit), 0.0)
            };
            // No normals array → leave each point's normal zero.
            let normal = match &data.normals_xyz {
                Some(n) => Vector3::new(n[[i, 0]], n[[i, 1]], n[[i, 2]]),
                None => Vector3::zeros(),
            };
            points.push(Point3D {
                position,
                w,
                color: [
                    data.colors_rgb[[i, 0]],
                    data.colors_rgb[[i, 1]],
                    data.colors_rgb[[i, 2]],
                ],
                error: data.reprojection_errors[i],
                normal,
            });
        }

        // Convert tracks (the mode-specific column is held in `observations`).
        let mut tracks = Vec::with_capacity(observation_count);
        for i in 0..observation_count {
            tracks.push(TrackObservation {
                image_index: data.image_indexes[i],
                point_index: data.point_indexes[i],
            });
        }

        // Build the observation-source columns from the mode-appropriate arrays.
        let observations = if is_embedded {
            ObservationSource::EmbeddedPatches {
                keypoints_xy: data.keypoints_xy.ok_or_else(|| {
                    SfmrError::InvalidFormat("embedded_patches file missing keypoints_xy".into())
                })?,
                image_file_hashes: data.image_file_hashes.ok_or_else(|| {
                    SfmrError::InvalidFormat(
                        "embedded_patches file missing image_file_hashes".into(),
                    )
                })?,
            }
        } else {
            ObservationSource::SiftFiles {
                feature_indexes: data
                    .feature_indexes
                    .ok_or_else(|| {
                        SfmrError::InvalidFormat("sift_files file missing feature_indexes".into())
                    })?
                    .to_vec(),
                feature_tool_hashes: data.feature_tool_hashes.ok_or_else(|| {
                    SfmrError::InvalidFormat("sift_files file missing feature_tool_hashes".into())
                })?,
                sift_content_hashes: data.sift_content_hashes.ok_or_else(|| {
                    SfmrError::InvalidFormat("sift_files file missing sift_content_hashes".into())
                })?,
            }
        };

        // Convert observation counts and compute prefix sum offsets
        let observation_counts = data.observation_counts.to_vec();
        let observation_offsets = compute_observation_offsets(&observation_counts);

        // Build per-image feature→point mapping and max feature index. These index
        // `.sift` features, so they are meaningful only for `sift_files`; an
        // `embedded_patches` reconstruction leaves them empty.
        let mut image_feature_to_point = vec![HashMap::new(); image_count];
        let mut max_track_feature_index = vec![0u32; image_count];
        if let ObservationSource::SiftFiles {
            feature_indexes, ..
        } = &observations
        {
            for (obs, &feat) in tracks.iter().zip(feature_indexes) {
                let img = obs.image_index as usize;
                image_feature_to_point[img].insert(feat, obs.point_index);
                max_track_feature_index[img] = max_track_feature_index[img].max(feat);
            }
        }

        // Convert depth histogram counts: (N, num_buckets) array → Vec<Vec<u32>>
        let mut depth_histogram_counts = Vec::with_capacity(image_count);
        for i in 0..image_count {
            let row: Vec<u32> = (0..num_buckets)
                .map(|j| data.observed_depth_histogram_counts[[i, j]])
                .collect();
            depth_histogram_counts.push(row);
        }

        // Convert SfmrCamera (serialization type) → CameraIntrinsics (computation type)
        let cameras: Vec<CameraIntrinsics> = data
            .cameras
            .iter()
            .map(|c| {
                CameraIntrinsics::try_from(c).map_err(|e| {
                    SfmrError::InvalidFormat(format!("invalid camera intrinsics: {e}"))
                })
            })
            .collect::<Result<Vec<_>, _>>()?;

        let infinity_point_count = count_points_at_infinity(&points);

        let recon = SfmrReconstruction {
            workspace_dir: data.workspace_dir.unwrap_or_default(),
            metadata: data.metadata,
            content_hash: data.content_hash,
            cameras,
            images,
            points,
            tracks,
            observation_counts,
            observation_offsets,
            thumbnails_y_x_rgb: data.thumbnails_y_x_rgb,
            depth_statistics: data.depth_statistics,
            depth_histogram_counts,
            rig_frame_data: data.rig_frame_data,
            patch_u_halfvec_xyz: data.patch_u_halfvec_xyz,
            patch_v_halfvec_xyz: data.patch_v_halfvec_xyz,
            patch_bitmaps_y_x_rgba: data.patch_bitmaps_y_x_rgba,
            has_normals,
            observations,
            image_feature_to_point,
            max_track_feature_index,
            infinity_point_count,
        };

        // The `load()` path is already protected by the format reader's
        // validation, but `from_sfmr_data` is also reached from the raw PyO3
        // `from_data` builder, so confirm the observation columns are parallel
        // before handing back a reconstruction.
        recon
            .validate_observation_columns()
            .map_err(SfmrError::InvalidFormat)?;
        Ok(recon)
    }

    /// Convert to the raw columnar I/O representation.
    pub fn to_sfmr_data(&self) -> SfmrData {
        use ndarray::{Array1, Array2};

        let image_count = self.images.len();
        let point_count = self.points.len();
        let observation_count = self.tracks.len();
        let num_buckets = self.depth_statistics.num_histogram_buckets as usize;
        // Keep the emitted metadata's discriminator in sync with the variant,
        // and its derived counts in sync with the actual arrays (in-memory
        // editors such as clone_with_changes can change the array sizes without
        // touching metadata).
        let mut metadata = self.metadata.clone();
        metadata.feature_source = self.feature_source().to_string();
        metadata.image_count = image_count as u32;
        metadata.point_count = point_count as u32;
        metadata.observation_count = observation_count as u32;
        metadata.infinity_point_count = self.infinity_point_count as u32;

        // Images
        let image_names: Vec<String> = self.images.iter().map(|im| im.name.clone()).collect();
        let mut camera_indexes = Array1::<u32>::zeros(image_count);
        let mut quaternions_wxyz = Array2::<f64>::zeros((image_count, 4));
        let mut translations_xyz = Array2::<f64>::zeros((image_count, 3));

        for (i, im) in self.images.iter().enumerate() {
            camera_indexes[i] = im.camera_index;
            let q = im.quaternion_wxyz.quaternion();
            quaternions_wxyz[[i, 0]] = q.w;
            quaternions_wxyz[[i, 1]] = q.i;
            quaternions_wxyz[[i, 2]] = q.j;
            quaternions_wxyz[[i, 3]] = q.k;
            translations_xyz[[i, 0]] = im.translation_xyz.x;
            translations_xyz[[i, 1]] = im.translation_xyz.y;
            translations_xyz[[i, 2]] = im.translation_xyz.z;
        }

        // Points. The ergonomic form is normalised — finite points have w = 1
        // and an Euclidean position, infinity points have w = 0 and a unit
        // direction — so the homogeneous row is just `(x, y, z, w)`.
        let mut positions_xyzw = Array2::<f64>::zeros((point_count, 4));
        let mut colors_rgb = Array2::<u8>::zeros((point_count, 3));
        let mut reprojection_errors = Array1::<f32>::zeros(point_count);
        // Normals are optional: `None` when this reconstruction carries none.
        let mut normals_xyz = self
            .has_normals
            .then(|| Array2::<f32>::zeros((point_count, 3)));

        for (i, pt) in self.points.iter().enumerate() {
            positions_xyzw[[i, 0]] = pt.position.x;
            positions_xyzw[[i, 1]] = pt.position.y;
            positions_xyzw[[i, 2]] = pt.position.z;
            positions_xyzw[[i, 3]] = pt.w;
            colors_rgb[[i, 0]] = pt.color[0];
            colors_rgb[[i, 1]] = pt.color[1];
            colors_rgb[[i, 2]] = pt.color[2];
            reprojection_errors[i] = pt.error;
            if let Some(normals) = &mut normals_xyz {
                normals[[i, 0]] = pt.normal.x;
                normals[[i, 1]] = pt.normal.y;
                normals[[i, 2]] = pt.normal.z;
            }
        }

        // Tracks (the mode-specific column comes from `observations` below).
        let mut image_indexes = Array1::<u32>::zeros(observation_count);
        let mut point_indexes = Array1::<u32>::zeros(observation_count);

        for (i, obs) in self.tracks.iter().enumerate() {
            image_indexes[i] = obs.image_index;
            point_indexes[i] = obs.point_index;
        }

        // Split the observation source back into the mode-appropriate columns.
        let (
            feature_indexes,
            feature_tool_hashes,
            sift_content_hashes,
            keypoints_xy,
            image_file_hashes,
        ) = match &self.observations {
            ObservationSource::SiftFiles {
                feature_indexes,
                feature_tool_hashes,
                sift_content_hashes,
            } => (
                Some(Array1::from_vec(feature_indexes.clone())),
                Some(feature_tool_hashes.clone()),
                Some(sift_content_hashes.clone()),
                None,
                None,
            ),
            ObservationSource::EmbeddedPatches {
                keypoints_xy,
                image_file_hashes,
            } => (
                None,
                None,
                None,
                Some(keypoints_xy.clone()),
                Some(image_file_hashes.clone()),
            ),
        };

        let observation_counts = Array1::from_vec(self.observation_counts.clone());

        // Depth histogram counts
        let mut observed_depth_histogram_counts = Array2::<u32>::zeros((image_count, num_buckets));
        for (i, row) in self.depth_histogram_counts.iter().enumerate() {
            for (j, &val) in row.iter().enumerate() {
                observed_depth_histogram_counts[[i, j]] = val;
            }
        }

        // Convert CameraIntrinsics → SfmrCamera for serialization
        let cameras: Vec<SfmrCamera> = self.cameras.iter().map(SfmrCamera::from).collect();

        SfmrData {
            workspace_dir: Some(self.workspace_dir.clone()),
            metadata,
            content_hash: self.content_hash.clone(),
            cameras,
            rig_frame_data: self.rig_frame_data.clone(),
            patch_u_halfvec_xyz: self.patch_u_halfvec_xyz.clone(),
            patch_v_halfvec_xyz: self.patch_v_halfvec_xyz.clone(),
            patch_bitmaps_y_x_rgba: self.patch_bitmaps_y_x_rgba.clone(),
            image_names,
            camera_indexes,
            quaternions_wxyz,
            translations_xyz,
            // Each mode emits only its own columns (computed above from the
            // ObservationSource variant).
            feature_tool_hashes,
            sift_content_hashes,
            image_file_hashes,
            thumbnails_y_x_rgb: self.thumbnails_y_x_rgb.clone(),
            positions_xyzw,
            colors_rgb,
            reprojection_errors,
            normals_xyz,
            image_indexes,
            feature_indexes,
            keypoints_xy,
            point_indexes,
            observation_counts,
            depth_statistics: self.depth_statistics.clone(),
            observed_depth_histogram_counts,
        }
    }
}
