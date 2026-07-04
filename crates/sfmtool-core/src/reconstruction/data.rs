// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Core `.sfmr` data types: [`SfmrReconstruction`] and its parts.
//!
//! [`SfmrReconstruction`] holds all data from a `.sfmr` file using nalgebra
//! geometric types. It converts to/from [`sfmr_format::SfmrData`] (the raw
//! columnar I/O representation) at the file I/O boundary.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use nalgebra::{Point3, Rotation3, UnitQuaternion, Vector3};
use ndarray::{Array2, Array4};

use sfmr_format::{
    resolve_workspace_dir, ContentHash, DepthStatistics, ImageDepthStats, ObservedDepthStats,
    RigFrameData, SfmrCamera, SfmrData, SfmrError, SfmrMetadata, FEATURE_SOURCE_EMBEDDED_PATCHES,
    FEATURE_SOURCE_SIFT_FILES,
};

use crate::camera::CameraIntrinsics;

/// Errors from reconstruction operations that require external data.
#[derive(Debug)]
pub enum ReconstructionError {
    /// Failed to read a `.sift` file.
    SiftRead { path: PathBuf, source: String },
    /// The operation is not supported for this reconstruction's feature source
    /// (e.g. a `.sift`-dependent step run on an `embedded_patches` recon).
    Unsupported(String),
}

impl std::fmt::Display for ReconstructionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ReconstructionError::SiftRead { path, source } => {
                write!(
                    f,
                    "failed to read SIFT file '{}': {}",
                    path.display(),
                    source
                )
            }
            ReconstructionError::Unsupported(msg) => write!(f, "{msg}"),
        }
    }
}

impl std::error::Error for ReconstructionError {}

/// A 3D point in the reconstruction.
///
/// Points are homogeneous: the on-disk `.sfmr` v2 format stores `(x, y, z, w)`.
/// Here the representation is normalised — a finite point (`w != 0`) stores its
/// Euclidean position in `position` with `w == 1.0`; a point at infinity
/// (`w == 0`) stores a unit-length direction in `position` with `w == 0.0`.
#[derive(Debug, Clone)]
pub struct Point3D {
    /// Euclidean position (finite point) or unit direction (point at infinity),
    /// in world coordinates. Disambiguated by `w`.
    pub position: Point3<f64>,
    /// Homogeneous coordinate kind: `1.0` for a finite point, `0.0` for a point
    /// at infinity.
    pub w: f64,
    /// RGB color (0-255 each).
    pub color: [u8; 3],
    /// RMS reprojection error in pixels.
    pub error: f32,
    /// Surface normal (unit vector in world coordinates). The default
    /// mean-viewing estimate leaves this `(0, 0, 0)` for a point at infinity.
    pub normal: Vector3<f32>,
}

impl Point3D {
    /// Whether this point is at infinity (`w == 0`).
    pub fn is_at_infinity(&self) -> bool {
        self.w == 0.0
    }
}

/// An image in the reconstruction with its pose.
#[derive(Debug, Clone)]
pub struct SfmrImage {
    /// Image path relative to workspace (POSIX format).
    pub name: String,
    /// Index into the cameras array.
    pub camera_index: u32,
    /// World-to-camera rotation quaternion (WXYZ).
    pub quaternion_wxyz: UnitQuaternion<f64>,
    /// World-to-camera translation vector.
    pub translation_xyz: Vector3<f64>,
}

impl SfmrImage {
    /// Compute the camera center in world coordinates.
    ///
    /// For world-to-camera transform `(R, t)`, the camera center is `C = -R^T * t`.
    pub fn camera_center(&self) -> Point3<f64> {
        let r = self.quaternion_wxyz.to_rotation_matrix();
        Point3::from(-(r.transpose() * self.translation_xyz))
    }

    /// Compute the camera-to-world rotation matrix as a row-major `[f64; 9]`.
    ///
    /// This is R^T where R is the world-to-camera rotation from `quaternion_wxyz`.
    /// The result is in the format expected by [`crate::camera::frustum::compute_frustum_corners`].
    pub fn camera_to_world_rotation_flat(&self) -> [f64; 9] {
        let r = self.quaternion_wxyz.inverse().to_rotation_matrix();
        let m = r.matrix();
        [
            m[(0, 0)],
            m[(0, 1)],
            m[(0, 2)],
            m[(1, 0)],
            m[(1, 1)],
            m[(1, 2)],
            m[(2, 0)],
            m[(2, 1)],
            m[(2, 2)],
        ]
    }
}

/// A single track observation linking an image to a 3D point.
///
/// The mode-specific 2D pixel — a `.sift` feature index (`sift_files`) or an
/// inline keypoint (`embedded_patches`) — lives in
/// [`SfmrReconstruction::observations`], a column parallel to the track array,
/// rather than in this struct (so neither mode pays for the other's field).
#[derive(Debug, Clone, Copy)]
pub struct TrackObservation {
    /// Index into the images array.
    pub image_index: u32,
    /// Index into the points array.
    pub point_index: u32,
}

/// The observation-source-specific columns of a reconstruction, selected once at
/// the array level (the file is wholly one mode — see "Observation source" in
/// `specs/formats/sfmr-file-format.md`). Each variant owns exactly its mode's
/// per-observation and per-image data, so neither carries placeholders for the
/// other.
#[derive(Debug, Clone)]
pub enum ObservationSource {
    /// Observations reference external `.sift` features.
    SiftFiles {
        /// `(M,)` feature index per observation, parallel to `tracks`.
        feature_indexes: Vec<u32>,
        /// XXH128 of the feature-extraction tool config, per image.
        feature_tool_hashes: Vec<[u8; 16]>,
        /// XXH128 of the `.sift` file content, per image.
        sift_content_hashes: Vec<[u8; 16]>,
    },
    /// Per-observation keypoints stored inline (no `.sift` companion).
    EmbeddedPatches {
        /// `(M, 2)` sub-pixel `(u, v)` per observation, parallel to `tracks`.
        keypoints_xy: Array2<f32>,
        /// XXH128 of the source image bytes, per image.
        image_file_hashes: Vec<[u8; 16]>,
    },
}

impl ObservationSource {
    /// The `feature_source` discriminator string for this variant.
    pub fn name(&self) -> &'static str {
        match self {
            ObservationSource::SiftFiles { .. } => FEATURE_SOURCE_SIFT_FILES,
            ObservationSource::EmbeddedPatches { .. } => FEATURE_SOURCE_EMBEDDED_PATCHES,
        }
    }
}

/// A full SfM reconstruction with all `.sfmr` data in ergonomic Rust types.
///
/// This is the Rust equivalent of Python's `SfmrReconstruction` class.
/// All fields from the `.sfmr` format are represented.
#[derive(Clone)]
pub struct SfmrReconstruction {
    /// Resolved workspace directory path.
    pub workspace_dir: PathBuf,
    /// Top-level reconstruction metadata.
    pub metadata: SfmrMetadata,
    /// Content integrity hashes (from the file, or empty if newly constructed).
    pub content_hash: ContentHash,
    /// Camera intrinsic parameters.
    pub cameras: Vec<CameraIntrinsics>,
    /// Registered images with poses.
    pub images: Vec<SfmrImage>,
    /// 3D points with colors, errors, and normals.
    pub points: Vec<Point3D>,
    /// Track observations (sorted by point_index, then image_index).
    pub tracks: Vec<TrackObservation>,
    /// Number of observations per 3D point.
    pub observation_counts: Vec<u32>,
    /// `(N, 128, 128, 3)` RGB thumbnails of the source images.
    pub thumbnails_y_x_rgb: Array4<u8>,
    /// Per-image depth statistics.
    pub depth_statistics: DepthStatistics,
    /// Depth histogram counts: `depth_histogram_counts[i]` has `num_histogram_buckets` entries.
    pub depth_histogram_counts: Vec<Vec<u32>>,
    /// Rig definitions and frame groupings. `None` when no multi-camera rigs.
    pub rig_frame_data: Option<RigFrameData>,
    /// Optional per-point oriented-patch frame (parallel to `points`), persisted
    /// in `points3d/` (version 3+). `patch_u_halfvec_xyz` and
    /// `patch_v_halfvec_xyz` are the in-plane half-extent vectors (both present
    /// or both `None`); a patch's center is its point's position and its normal
    /// is the point's `normal`. See [`crate::patch::PatchCloud`].
    pub patch_u_halfvec_xyz: Option<Array2<f32>>,
    pub patch_v_halfvec_xyz: Option<Array2<f32>>,
    /// Optional `(P, R, R, 4)` per-point RGBA patch bitmaps; the alpha channel
    /// holds a per-pixel confidence.
    pub patch_bitmaps_y_x_rgba: Option<Array4<u8>>,
    /// Whether this reconstruction carries per-point normals. When `false`, each
    /// point's inline `normal` is left zero and the columnar `normals_xyz` array
    /// is neither built nor written. `true` for everything loaded from versions 1
    /// and 2.
    pub has_normals: bool,
    /// The observation-source-specific columns (per-observation feature index or
    /// keypoint, per-image hashes), selected by variant. The feature→point maps
    /// below are meaningful only for [`ObservationSource::SiftFiles`].
    pub observations: ObservationSource,

    // --- Derived data (computed from the fields above, not stored in .sfmr) ---
    /// Prefix sum of `observation_counts`: `observation_offsets[i]` is the
    /// index into `tracks` where point `i`'s observations begin.
    /// Length: `points.len() + 1` (last element = total observation count).
    pub observation_offsets: Vec<usize>,
    /// Per-image mapping from feature_index → point_index for tracked features.
    /// Outer vec indexed by image_index.
    pub image_feature_to_point: Vec<HashMap<u32, u32>>,
    /// Max feature_index referenced by any track observation for each image.
    /// Used to determine how many features to read from the .sift file.
    pub max_track_feature_index: Vec<u32>,
    /// Cached count of 3D points at infinity (`w == 0`). Refreshed by
    /// `rebuild_derived_fields` and by the in-place `w`-mutators
    /// (`classify_points_at_infinity` / `materialize_points_at_infinity`), since
    /// the count depends on point `w`-values rather than the track structure the
    /// other derived fields track.
    pub infinity_point_count: usize,
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

    /// The `feature_source` discriminator (`"sift_files"` / `"embedded_patches"`).
    pub fn feature_source(&self) -> &str {
        self.observations.name()
    }

    /// Per-observation feature indexes (parallel to `tracks`), or `None` for an
    /// `embedded_patches` reconstruction.
    pub fn feature_indexes(&self) -> Option<&[u32]> {
        match &self.observations {
            ObservationSource::SiftFiles {
                feature_indexes, ..
            } => Some(feature_indexes),
            ObservationSource::EmbeddedPatches { .. } => None,
        }
    }

    /// Per-observation sub-pixel keypoints `(M, 2)`, or `None` for a `sift_files`
    /// reconstruction.
    pub fn keypoints_xy(&self) -> Option<&Array2<f32>> {
        match &self.observations {
            ObservationSource::EmbeddedPatches { keypoints_xy, .. } => Some(keypoints_xy),
            ObservationSource::SiftFiles { .. } => None,
        }
    }

    /// Derive the local **affine shape** of an observation's keypoint — the
    /// counterpart of the `.sift` affine frame carried by `sift_files` — by
    /// projecting the point's patch frame `(u, v)` into the observing camera.
    /// The returned 2×2 matrix has the projected patch half-axes as its columns,
    /// so (like a `.sift` shape) it maps the unit circle to the keypoint's image
    /// footprint: an overlay recovers scale from the column norms, orientation
    /// from their angle, and anisotropy from their ratio.
    ///
    /// `keypoint_xy` anchors the frame in-plane (the patch-plane point that
    /// projects to it); a grazing view falls back to the point centre. A point
    /// at infinity is handled too: its patch is tangent to the direction sphere,
    /// so the frame is evaluated at the stored direction and its corners project
    /// as directions (the camera translation folds out), giving a roughly
    /// circular shape. Returns `None` when the reconstruction carries no patch
    /// frame, this point has no patch (zero frame), or the projection is
    /// degenerate or behind the camera. See `specs/formats/sfmr-file-format.md`
    /// ("Deriving keypoint shape, scale, and orientation").
    pub fn observation_affine_shape(
        &self,
        point_idx: usize,
        image_index: usize,
        keypoint_xy: [f32; 2],
    ) -> Option<[[f32; 2]; 2]> {
        let u_arr = self.patch_u_halfvec_xyz.as_ref()?;
        let v_arr = self.patch_v_halfvec_xyz.as_ref()?;
        let point = self.points.get(point_idx)?;
        let u = Vector3::new(
            u_arr[[point_idx, 0]] as f64,
            u_arr[[point_idx, 1]] as f64,
            u_arr[[point_idx, 2]] as f64,
        );
        let v = Vector3::new(
            v_arr[[point_idx, 0]] as f64,
            v_arr[[point_idx, 1]] as f64,
            v_arr[[point_idx, 2]] as f64,
        );
        if u.norm_squared() == 0.0 || v.norm_squared() == 0.0 {
            return None; // no patch for this point
        }

        let image = self.images.get(image_index)?;
        let camera = self.cameras.get(image.camera_index as usize)?;
        let r = image.quaternion_wxyz.to_rotation_matrix();

        // Where to evaluate the frame. For a point at infinity the patch is
        // tangent to the direction sphere: the anchor is the stored direction
        // and corners are directions (projected with `w = 0`). For a finite
        // point the anchor is where the keypoint's back-projected ray meets the
        // patch plane (fall back to the point centre for a grazing view).
        let anchor = if point.is_at_infinity() {
            point.position.coords
        } else {
            let normal = u.cross(&v);
            let n_norm = normal.norm();
            if n_norm == 0.0 {
                return None;
            }
            let normal = normal / n_norm;
            let center = image.camera_center();
            let ray_cam = camera.pixel_to_ray(keypoint_xy[0] as f64, keypoint_xy[1] as f64);
            let ray_world = r.transpose() * Vector3::new(ray_cam[0], ray_cam[1], ray_cam[2]);
            let denom = ray_world.dot(&normal);
            if denom.abs() < 1e-9 {
                point.position.coords
            } else {
                let lambda = (point.position.coords - center.coords).dot(&normal) / denom;
                center.coords + lambda * ray_world
            }
        };

        // Project the anchor and the two half-axis tips. `w` folds out the
        // translation for a point at infinity, so its corners project as
        // directions.
        let w = point.w;
        let project = |world: Vector3<f64>| -> Option<(f64, f64)> {
            let p_cam = r * world + w * image.translation_xyz;
            camera.ray_to_pixel([p_cam.x, p_cam.y, p_cam.z])
        };
        let k = project(anchor)?;
        let pu = project(anchor + u)?;
        let pv = project(anchor + v)?;

        // Columns are the projected half-axes: u -> column 0, v -> column 1.
        Some([
            [(pu.0 - k.0) as f32, (pv.0 - k.0) as f32],
            [(pu.1 - k.1) as f32, (pv.1 - k.1) as f32],
        ])
    }

    /// Per-image feature-tool hashes, or `None` for `embedded_patches`.
    pub fn feature_tool_hashes(&self) -> Option<&[[u8; 16]]> {
        match &self.observations {
            ObservationSource::SiftFiles {
                feature_tool_hashes,
                ..
            } => Some(feature_tool_hashes),
            ObservationSource::EmbeddedPatches { .. } => None,
        }
    }

    /// Per-image `.sift`-content hashes, or `None` for `embedded_patches`.
    pub fn sift_content_hashes(&self) -> Option<&[[u8; 16]]> {
        match &self.observations {
            ObservationSource::SiftFiles {
                sift_content_hashes,
                ..
            } => Some(sift_content_hashes),
            ObservationSource::EmbeddedPatches { .. } => None,
        }
    }

    /// Per-image source-image hashes, or `None` for `sift_files`.
    pub fn image_file_hashes(&self) -> Option<&[[u8; 16]]> {
        match &self.observations {
            ObservationSource::EmbeddedPatches {
                image_file_hashes, ..
            } => Some(image_file_hashes),
            ObservationSource::SiftFiles { .. } => None,
        }
    }

    /// Check that the observation-source columns are parallel to the structures
    /// they annotate: per-observation columns (`feature_indexes` / `keypoints_xy`)
    /// must match the track count, and per-image columns (the hashes) must match
    /// the image count. Returns an error message describing the first mismatch.
    ///
    /// `from_sfmr_data` builds these in lockstep, but the in-memory editors
    /// (notably `clone_with_changes`, which can replace tracks and columns
    /// independently) can leave them out of step; this is the guard those paths
    /// run before handing back a reconstruction.
    pub fn validate_observation_columns(&self) -> Result<(), String> {
        let n_obs = self.tracks.len();
        let n_img = self.images.len();
        match &self.observations {
            ObservationSource::SiftFiles {
                feature_indexes,
                feature_tool_hashes,
                sift_content_hashes,
            } => {
                if feature_indexes.len() != n_obs {
                    return Err(format!(
                        "feature_indexes length ({}) must match observation count ({n_obs})",
                        feature_indexes.len()
                    ));
                }
                if feature_tool_hashes.len() != n_img {
                    return Err(format!(
                        "feature_tool_hashes length ({}) must match image count ({n_img})",
                        feature_tool_hashes.len()
                    ));
                }
                if sift_content_hashes.len() != n_img {
                    return Err(format!(
                        "sift_content_hashes length ({}) must match image count ({n_img})",
                        sift_content_hashes.len()
                    ));
                }
            }
            ObservationSource::EmbeddedPatches {
                keypoints_xy,
                image_file_hashes,
            } => {
                if keypoints_xy.nrows() != n_obs {
                    return Err(format!(
                        "keypoints_xy row count ({}) must match observation count ({n_obs})",
                        keypoints_xy.nrows()
                    ));
                }
                if image_file_hashes.len() != n_img {
                    return Err(format!(
                        "image_file_hashes length ({}) must match image count ({n_img})",
                        image_file_hashes.len()
                    ));
                }
            }
        }
        Ok(())
    }

    /// Number of registered images.
    pub fn image_count(&self) -> usize {
        self.images.len()
    }

    /// Number of 3D points.
    pub fn point_count(&self) -> usize {
        self.points.len()
    }

    /// Number of track observations.
    pub fn observation_count(&self) -> usize {
        self.tracks.len()
    }

    /// Number of camera models.
    pub fn camera_count(&self) -> usize {
        self.cameras.len()
    }

    /// Return the observations for a given 3D point. O(1) lookup.
    pub fn observations_for_point(&self, point_idx: usize) -> &[TrackObservation] {
        let start = self.observation_offsets[point_idx];
        let end = self.observation_offsets[point_idx + 1];
        &self.tracks[start..end]
    }

    /// Compute per-observation reprojection errors for a single image.
    ///
    /// Loads feature positions from the image's `.sift` file, projects each
    /// observed 3D point through the camera, and measures pixel distance to
    /// the observed feature position. Points at infinity (`w = 0`) project
    /// their stored bearing direction (translation-free), so their error is
    /// well-defined like any finite point's.
    ///
    /// Returns a vector of `(feature_index, reprojection_error_px)` pairs,
    /// one per track observation for this image. Points behind the camera
    /// produce `f32::NAN`.
    pub fn compute_observation_reprojection_errors(
        &self,
        image_index: usize,
    ) -> Result<Vec<(u32, f32)>, ReconstructionError> {
        let image = &self.images[image_index];
        let camera = &self.cameras[image.camera_index as usize];

        // Determine how many features we need from the sift file
        let max_feat_idx = self.max_track_feature_index[image_index] as usize;
        let read_count = max_feat_idx + 1;

        // Load feature positions from the .sift file
        let sift_path = self.sift_path_for_image(image_index);
        let positions = sift_format::read_sift_positions(&sift_path, read_count).map_err(|e| {
            ReconstructionError::SiftRead {
                path: sift_path,
                source: e.to_string(),
            }
        })?;

        // World-to-camera rotation matrix
        let r = image.quaternion_wxyz.to_rotation_matrix();
        let t = &image.translation_xyz;

        // Iterate all track observations for this image
        let feat_to_point = &self.image_feature_to_point[image_index];
        let mut results = Vec::with_capacity(feat_to_point.len());

        for (&feat_idx, &point_idx) in feat_to_point {
            let feature_xy = match positions.get(feat_idx as usize) {
                Some(&xy) => xy,
                None => {
                    results.push((feat_idx, f32::NAN));
                    continue;
                }
            };

            let point = &self.points[point_idx as usize];
            let observed = [feature_xy[0] as f64, feature_xy[1] as f64];
            let error = match observation_reprojection_error(
                &r,
                t,
                camera,
                &point.position,
                point.is_at_infinity(),
                observed,
            ) {
                Some(e) => e as f32,
                // Point behind the camera: no defined reprojection.
                None => f32::NAN,
            };

            results.push((feat_idx, error));
        }

        Ok(results)
    }

    /// Recompute per-point mean reprojection errors from scratch.
    ///
    /// For each image, loads feature positions from the `.sift` file and
    /// reprojects all observed 3D points through the camera model. Each
    /// point's `error` is set to the mean pixel-space reprojection error
    /// across all its observations. Points with no valid observations
    /// (e.g., all behind camera) get `error = 0.0`.
    ///
    /// This replaces any errors read from COLMAP/GLOMAP binary files,
    /// which may use different coordinate conventions (GLOMAP stores
    /// errors in normalized image coordinates, not pixels).
    pub fn recompute_point_errors(&mut self) -> Result<(), ReconstructionError> {
        let num_points = self.points.len();
        let mut error_sums = vec![0.0f64; num_points];
        let mut error_counts = vec![0u32; num_points];

        for img_idx in 0..self.images.len() {
            let results = self.compute_observation_reprojection_errors(img_idx)?;
            let feat_to_point = &self.image_feature_to_point[img_idx];
            for (feat_idx, error) in results {
                if error.is_nan() {
                    continue;
                }
                if let Some(&point_idx) = feat_to_point.get(&feat_idx) {
                    error_sums[point_idx as usize] += error as f64;
                    error_counts[point_idx as usize] += 1;
                }
            }
        }

        for i in 0..num_points {
            self.points[i].error = if error_counts[i] > 0 {
                (error_sums[i] / error_counts[i] as f64) as f32
            } else {
                0.0
            };
        }

        Ok(())
    }

    /// Recompute mean reprojection errors for points at infinity only, leaving
    /// finite points' errors untouched.
    ///
    /// Used after bundle adjustment: a point that was materialised to a finite
    /// landmark, refined, then reclassified back to `w = 0` carries an error
    /// describing the landmark, not its bearing. Only those points need fixing,
    /// so finite points keep the errors the solve produced and `.sift` files are
    /// read only for images that observe a point at infinity.
    pub fn recompute_infinity_point_errors(&mut self) -> Result<(), ReconstructionError> {
        let num_points = self.points.len();
        let is_infinity: Vec<bool> = self.points.iter().map(|p| p.is_at_infinity()).collect();
        if !is_infinity.iter().any(|&b| b) {
            return Ok(());
        }

        let mut error_sums = vec![0.0f64; num_points];
        let mut error_counts = vec![0u32; num_points];

        for img_idx in 0..self.images.len() {
            let feat_to_point = &self.image_feature_to_point[img_idx];
            // Skip images observing no point at infinity — avoids reading their
            // `.sift` file just to discard every observation.
            if !feat_to_point.values().any(|&p| is_infinity[p as usize]) {
                continue;
            }
            let results = self.compute_observation_reprojection_errors(img_idx)?;
            let feat_to_point = &self.image_feature_to_point[img_idx];
            for (feat_idx, error) in results {
                if error.is_nan() {
                    continue;
                }
                if let Some(&point_idx) = feat_to_point.get(&feat_idx) {
                    if is_infinity[point_idx as usize] {
                        error_sums[point_idx as usize] += error as f64;
                        error_counts[point_idx as usize] += 1;
                    }
                }
            }
        }

        for i in 0..num_points {
            if is_infinity[i] {
                self.points[i].error = if error_counts[i] > 0 {
                    (error_sums[i] / error_counts[i] as f64) as f32
                } else {
                    0.0
                };
            }
        }

        Ok(())
    }

    /// Return the image indices that observe a given 3D point.
    pub fn track_image_indices(&self, point_idx: usize) -> Vec<usize> {
        self.observations_for_point(point_idx)
            .iter()
            .map(|obs| obs.image_index as usize)
            .collect()
    }

    /// Return the expected `.sift` file path for a given image index.
    ///
    /// The path follows the convention:
    /// `{workspace_dir}/{image_parent}/{feature_prefix_dir}/{image_basename}.sift`
    ///
    /// where `feature_prefix_dir` (e.g., `"features/sift-colmap-{hash}"`) is
    /// stored in `metadata.workspace.feature_prefix_dir`.
    pub fn sift_path_for_image(&self, image_idx: usize) -> PathBuf {
        let prefix = &self.metadata.workspace.contents.feature_prefix_dir;
        let image = &self.images[image_idx];
        let image_rel = Path::new(&image.name);
        let image_parent = image_rel.parent().unwrap_or(Path::new(""));
        let image_basename = image_rel.file_name().unwrap_or_default();
        let sift_filename = format!("{}.sift", image_basename.to_string_lossy());

        self.workspace_dir
            .join(image_parent)
            .join(prefix)
            .join(&sift_filename)
    }

    /// Recompute depth statistics, histograms, and estimated normals from the
    /// current poses, points, and tracks. Uses the same
    /// [`sfmr_format::compute_depth_statistics`] function that `.sfmr` file
    /// writing uses.
    pub fn recompute_depth_statistics(&mut self) -> Result<(), SfmrError> {
        use ndarray::{Array1, Array2};

        let image_count = self.images.len();
        let points3d_count = self.points.len();
        let observation_count = self.tracks.len();

        // Build columnar arrays from the reconstruction data
        let mut quaternions_wxyz = Array2::<f64>::zeros((image_count, 4));
        let mut translations_xyz = Array2::<f64>::zeros((image_count, 3));
        for (i, im) in self.images.iter().enumerate() {
            let q = im.quaternion_wxyz.quaternion();
            quaternions_wxyz[[i, 0]] = q.w;
            quaternions_wxyz[[i, 1]] = q.i;
            quaternions_wxyz[[i, 2]] = q.j;
            quaternions_wxyz[[i, 3]] = q.k;
            translations_xyz[[i, 0]] = im.translation_xyz.x;
            translations_xyz[[i, 1]] = im.translation_xyz.y;
            translations_xyz[[i, 2]] = im.translation_xyz.z;
        }

        let mut positions_xyzw = Array2::<f64>::zeros((points3d_count, 4));
        for (i, pt) in self.points.iter().enumerate() {
            positions_xyzw[[i, 0]] = pt.position.x;
            positions_xyzw[[i, 1]] = pt.position.y;
            positions_xyzw[[i, 2]] = pt.position.z;
            positions_xyzw[[i, 3]] = pt.w;
        }

        let mut image_indexes = Array1::<u32>::zeros(observation_count);
        let mut point_indexes = Array1::<u32>::zeros(observation_count);
        for (i, obs) in self.tracks.iter().enumerate() {
            image_indexes[i] = obs.image_index;
            point_indexes[i] = obs.point_index;
        }

        let result = sfmr_format::compute_depth_statistics(
            &quaternions_wxyz,
            &translations_xyz,
            &positions_xyzw,
            &image_indexes,
            &point_indexes,
        )?;

        // Store results back
        self.depth_statistics = result.depth_statistics;
        let num_buckets = result.observed_depth_histogram_counts.ncols();
        self.depth_histogram_counts = (0..image_count)
            .map(|i| {
                (0..num_buckets)
                    .map(|j| result.observed_depth_histogram_counts[[i, j]])
                    .collect()
            })
            .collect();
        // Only materialize normals when this reconstruction carries them.
        if self.has_normals {
            for (i, pt) in self.points.iter_mut().enumerate() {
                pt.normal = Vector3::new(
                    result.mean_viewing_normals_xyz[[i, 0]],
                    result.mean_viewing_normals_xyz[[i, 1]],
                    result.mean_viewing_normals_xyz[[i, 2]],
                );
            }
        }

        Ok(())
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
            let quaternion =
                UnitQuaternion::new_normalize(nalgebra::Quaternion::new(qw, qx, qy, qz));

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

    /// Rebuild derived fields (observation offsets, feature→point maps, and the
    /// `infinity_point_count` cache) from the current `tracks`,
    /// `observation_counts`, `images`, and `points`.
    ///
    /// Call this after mutating tracks, observation counts, or point
    /// `w`-values externally.
    pub fn rebuild_derived_fields(&mut self) {
        self.observation_offsets = compute_observation_offsets(&self.observation_counts);

        let image_count = self.images.len();
        self.image_feature_to_point = vec![HashMap::new(); image_count];
        self.max_track_feature_index = vec![0u32; image_count];
        if let ObservationSource::SiftFiles {
            feature_indexes, ..
        } = &self.observations
        {
            for (obs, &feat) in self.tracks.iter().zip(feature_indexes) {
                let img = obs.image_index as usize;
                self.image_feature_to_point[img].insert(feat, obs.point_index);
                self.max_track_feature_index[img] = self.max_track_feature_index[img].max(feat);
            }
        }

        self.infinity_point_count = count_points_at_infinity(&self.points);
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

    /// Creates a demo reconstruction with synthetic data for testing.
    ///
    /// Generates `num_points` 3D points evenly distributed on a unit sphere
    /// (offset to sit in front of the camera arc), observed by 8 cameras
    /// arranged in a circle around the origin.
    pub fn demo(num_points: usize) -> Self {
        use crate::camera::CameraModel;
        use crate::spherical::sphere_points::{evenly_distributed_sphere_points, RelaxConfig};
        use std::collections::HashMap;

        let num_images = 8;
        let num_buckets: u32 = 128;

        // Camera
        let cameras = vec![CameraIntrinsics {
            model: CameraModel::Pinhole {
                focal_length_x: 1000.0,
                focal_length_y: 1000.0,
                principal_point_x: 960.0,
                principal_point_y: 540.0,
            },
            width: 1920,
            height: 1080,
        }];

        // Images: cameras in an arc
        let mut images = Vec::with_capacity(num_images);
        for i in 0..num_images {
            let angle = (i as f64) * std::f64::consts::PI / 4.0;
            let radius = 5.0;
            let position = Point3::new(radius * angle.cos(), radius * angle.sin(), 1.5);

            // Look at origin: forward = -position.normalize()
            let forward = (Point3::origin() - position).normalize();
            let world_up = Vector3::z();
            let right = forward.cross(&world_up).normalize();
            let up = right.cross(&forward).normalize();

            // Build rotation matrix (world-to-camera, canonical convention:
            // camera looks down −Z with +Y up). Rows are the camera axes
            // expressed in world coordinates: X=right, Y=up, Z=−forward.
            let r = nalgebra::Matrix3::new(
                right.x, right.y, right.z, up.x, up.y, up.z, -forward.x, -forward.y, -forward.z,
            );
            let rotation = UnitQuaternion::from_rotation_matrix(
                &nalgebra::Rotation3::from_matrix_unchecked(r),
            );
            let translation = rotation * (-position.coords);

            images.push(SfmrImage {
                name: format!("image_{:03}.jpg", i),
                camera_index: 0,
                quaternion_wxyz: rotation,
                translation_xyz: translation,
            });
        }

        // Points: evenly distributed on the unit sphere via Thomson relaxation,
        // then offset by +1 in z so they sit in front of the camera arc.
        let sphere = evenly_distributed_sphere_points(num_points, &RelaxConfig::default());
        let mut points = Vec::with_capacity(num_points);
        for i in 0..num_points {
            let x = sphere[3 * i] as f64;
            let y = sphere[3 * i + 1] as f64;
            let z = sphere[3 * i + 2] as f64;

            let r = ((x + 1.0) * 127.5) as u8;
            let g = ((y + 1.0) * 127.5) as u8;
            let b = ((z + 1.0) * 127.5) as u8;

            points.push(Point3D {
                position: Point3::new(x, y, z + 1.0),
                w: 1.0,
                color: [r, g, b],
                error: 0.5 + (i as f64 / num_points.max(1) as f64) as f32 * 0.5,
                normal: Vector3::new(x as f32, y as f32, z as f32).normalize(),
            });
        }

        // Simple tracks: each point observed by 2 adjacent cameras. The parallel
        // feature_indexes column goes into the sift_files ObservationSource.
        let mut tracks = Vec::new();
        let mut feature_indexes = Vec::new();
        let mut observation_counts = Vec::with_capacity(num_points);
        for i in 0..num_points {
            let cam1 = (i % num_images) as u32;
            let cam2 = ((i + 1) % num_images) as u32;
            let (first, second) = if cam1 <= cam2 {
                (cam1, cam2)
            } else {
                (cam2, cam1)
            };
            tracks.push(TrackObservation {
                image_index: first,
                point_index: i as u32,
            });
            tracks.push(TrackObservation {
                image_index: second,
                point_index: i as u32,
            });
            feature_indexes.push(i as u32);
            feature_indexes.push(i as u32);
            observation_counts.push(2);
        }

        // Empty depth statistics
        let depth_statistics = DepthStatistics {
            num_histogram_buckets: num_buckets,
            images: (0..num_images)
                .map(|_| ImageDepthStats {
                    histogram_min_z: None,
                    histogram_max_z: None,
                    observed: ObservedDepthStats {
                        count: 0,
                        infinity_count: 0,
                        min_z: None,
                        max_z: None,
                        median_z: None,
                        mean_z: None,
                    },
                })
                .collect(),
        };
        let depth_histogram_counts = vec![vec![0u32; num_buckets as usize]; num_images];

        let metadata = SfmrMetadata {
            version: 2,
            operation: "demo".into(),
            tool: "sfmtool".into(),
            tool_version: "0.1.0".into(),
            tool_options: HashMap::new(),
            workspace: sfmr_format::WorkspaceMetadata {
                absolute_path: String::new(),
                relative_path: ".".into(),
                contents: sfmr_format::WorkspaceContents {
                    feature_tool: "none".into(),
                    feature_type: "sift".into(),
                    feature_options: serde_json::json!({}),
                    feature_prefix_dir: String::new(),
                },
            },
            timestamp: String::new(),
            image_count: num_images as u32,
            point_count: num_points as u32,
            infinity_point_count: 0,
            observation_count: (num_points * 2) as u32,
            camera_count: 1,
            rig_count: None,
            sensor_count: None,
            frame_count: None,
            world_space_unit: None,
            feature_source: FEATURE_SOURCE_SIFT_FILES.to_string(),
        };

        let observation_offsets = compute_observation_offsets(&observation_counts);

        // Build per-image feature→point mapping
        let mut image_feature_to_point = vec![HashMap::new(); num_images];
        let mut max_track_feature_index = vec![0u32; num_images];
        for (obs, &feat) in tracks.iter().zip(&feature_indexes) {
            let img = obs.image_index as usize;
            image_feature_to_point[img].insert(feat, obs.point_index);
            max_track_feature_index[img] = max_track_feature_index[img].max(feat);
        }

        let infinity_point_count = count_points_at_infinity(&points);
        SfmrReconstruction {
            infinity_point_count,
            workspace_dir: PathBuf::new(),
            metadata,
            rig_frame_data: None,
            patch_u_halfvec_xyz: None,
            patch_v_halfvec_xyz: None,
            patch_bitmaps_y_x_rgba: None,
            has_normals: true,
            observations: ObservationSource::SiftFiles {
                feature_indexes,
                feature_tool_hashes: vec![[0u8; 16]; num_images],
                sift_content_hashes: vec![[0u8; 16]; num_images],
            },
            content_hash: ContentHash {
                metadata_xxh128: String::new(),
                cameras_xxh128: String::new(),
                rigs_xxh128: None,
                frames_xxh128: None,
                images_xxh128: String::new(),
                points3d_xxh128: String::new(),
                tracks_xxh128: String::new(),
                content_xxh128: String::new(),
            },
            cameras,
            images,
            points,
            tracks,
            observation_counts,
            observation_offsets,
            thumbnails_y_x_rgb: Array4::zeros((num_images, 128, 128, 3)),
            depth_statistics,
            depth_histogram_counts,
            image_feature_to_point,
            max_track_feature_index,
        }
    }
}

/// Count 3D points at infinity (`w == 0`) in a slice. Shared by every
/// constructor and refresh site so the cached `infinity_point_count` stays
/// consistent.
pub(crate) fn count_points_at_infinity(points: &[Point3D]) -> usize {
    points.iter().filter(|p| p.is_at_infinity()).count()
}

/// Reprojection error (pixels) of one observation, or `None` if the point is
/// behind the camera.
///
/// `rotation`/`translation` are the world→camera pose; `point` is the 3D point's
/// stored coordinate. A point at infinity (`at_infinity`) projects its bearing
/// direction through rotation + intrinsics only — the camera translation is
/// negligible at infinity — while a finite point projects `R·p + t`. Shared by
/// per-image error computation and points-at-infinity discovery so the two stay
/// in sync.
pub(crate) fn observation_reprojection_error(
    rotation: &Rotation3<f64>,
    translation: &Vector3<f64>,
    camera: &CameraIntrinsics,
    point: &Point3<f64>,
    at_infinity: bool,
    observed_xy: [f64; 2],
) -> Option<f64> {
    let p_cam = if at_infinity {
        rotation * point.coords
    } else {
        rotation * point.coords + translation
    };
    // Cheirality: a point in front of a canonical camera has z < 0.
    if p_cam.z >= 0.0 {
        return None;
    }
    let (u_proj, v_proj) = camera.project(p_cam.x / -p_cam.z, p_cam.y / -p_cam.z);
    let du = u_proj - observed_xy[0];
    let dv = v_proj - observed_xy[1];
    Some((du * du + dv * dv).sqrt())
}

/// Compute prefix sum offsets from observation counts.
///
/// Returns a vector of length `counts.len() + 1` where `offsets[i]` is the
/// index into the tracks array where point `i`'s observations begin.
pub(super) fn compute_observation_offsets(counts: &[u32]) -> Vec<usize> {
    let mut offsets = Vec::with_capacity(counts.len() + 1);
    offsets.push(0);
    for &count in counts {
        offsets.push(offsets.last().unwrap() + count as usize);
    }
    offsets
}

#[cfg(test)]
mod tests;
