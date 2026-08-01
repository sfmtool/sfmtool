// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Core `.sfmr` data types: [`SfmrReconstruction`] and its parts.
//!
//! [`SfmrReconstruction`] holds all data from a `.sfmr` file using nalgebra
//! geometric types. This file owns the type definitions and their accessors;
//! the rest of the type's surface lives in three children:
//!
//! - [`conversion`] — the `.sfmr` boundary: the [`sfmr_format::SfmrData`] round
//!   trip (the raw columnar I/O representation) plus the `load`/`save` wrappers.
//! - [`recompute`] — derived quantities recomputed from geometry: per-observation
//!   and per-point reprojection errors, and the depth statistics/histograms.
//! - [`demo`] — a synthetic reconstruction with no files behind it.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use nalgebra::{Point3, UnitQuaternion, Vector3};
use ndarray::{Array2, Array4};

use sfmr_format::{
    ContentHash, DepthStatistics, RigFrameData, SfmrMetadata, FEATURE_SOURCE_EMBEDDED_PATCHES,
    FEATURE_SOURCE_SIFT_FILES,
};

use crate::camera::CameraIntrinsics;

mod conversion;
mod demo;
mod recompute;

// Re-exported at the old path: `analysis::infinity::discover` imports it as
// `crate::reconstruction::data::observation_reprojection_error`.
pub(crate) use recompute::observation_reprojection_error;

pub use conversion::unit_quaternion_preserving;

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

    /// Per-point maximum keypoint feature size (px) for an `embedded_patches`
    /// reconstruction, derived from each observation's projected patch frame.
    ///
    /// For every observation the point's patch frame is projected into the
    /// observing camera (`observation_affine_shape`) and its size taken as the
    /// mean of the two projected half-axis column norms — the same size measure
    /// the Track View reports and that `.sift` affine shapes yield for a
    /// `sift_files` reconstruction. The per-point value is the maximum over its
    /// observations. Returns `None` for a `sift_files` source (no inline
    /// keypoints); a point with no patch, or whose every observation projects
    /// degenerately, gets `0.0`.
    pub fn max_embedded_feature_size_per_point(&self) -> Option<Vec<f32>> {
        let keypoints_xy = self.keypoints_xy()?;
        let mut out = vec![0.0f32; self.point_count()];
        for (point_idx, slot) in out.iter_mut().enumerate() {
            let start = self.observation_offsets[point_idx];
            let mut max_size = 0.0f32;
            for (k, obs) in self.observations_for_point(point_idx).iter().enumerate() {
                let obs_global = start + k;
                let xy = [keypoints_xy[[obs_global, 0]], keypoints_xy[[obs_global, 1]]];
                if let Some(a) =
                    self.observation_affine_shape(point_idx, obs.image_index as usize, xy)
                {
                    let col0 = (a[0][0] * a[0][0] + a[1][0] * a[1][0]).sqrt();
                    let col1 = (a[0][1] * a[0][1] + a[1][1] * a[1][1]).sqrt();
                    max_size = max_size.max(0.5 * (col0 + col1));
                }
            }
            *slot = max_size;
        }
        Some(out)
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
}

/// Count 3D points at infinity (`w == 0`) in a slice. Shared by every
/// constructor and refresh site so the cached `infinity_point_count` stays
/// consistent.
pub(crate) fn count_points_at_infinity(points: &[Point3D]) -> usize {
    points.iter().filter(|p| p.is_at_infinity()).count()
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
