// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The point side of a reconstruction: the 3D points, the tracks that observe
//! them, and everything measured per point or per observation.

use std::collections::HashMap;
use std::sync::Arc;

use ndarray::{Array2, Array4};

use sfmr_format::{FEATURE_SOURCE_EMBEDDED_PATCHES, FEATURE_SOURCE_SIFT_FILES};

use super::PointConstraintColumns;
use super::{compute_observation_offsets, count_points_at_infinity, Point3D, TrackObservation};

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
        /// Optional `(M, 2)` inline `(u, v)` per observation, parallel to
        /// `tracks`. When present it is this reconstruction's own statement of
        /// where each observation sits, and every consumer that resolves an
        /// observation's pixel reads it in preference to the `.sift` feature the
        /// matching `feature_indexes` entry points at -- a producer may have
        /// refined the coordinate past the original detection. The `.sift` files
        /// stay the source of the feature itself (descriptor, scale, affine
        /// shape).
        keypoints_xy: Option<Array2<f32>>,
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

/// The 3D points, the tracks that observe them, and every column measured per
/// point or per observation -- one half of a
/// [`SfmrReconstruction`](super::SfmrReconstruction), the half an image never
/// belongs to.
///
/// The tracks are one CSR structure: `tracks` sorted by point index then image
/// index, `observation_counts` giving each point's run length, and
/// `observation_offsets` their prefix sum. Every per-observation column
/// (`observations`' pixel column, `observation_confidence`) is parallel to
/// `tracks`; every per-point column (`normal_confidence`, the patch frame and
/// bitmaps, `point_constraints`) is parallel to `points`.
///
/// The derived fields at the bottom are a function of the rest and of the image
/// count, and [`Self::rebuild_derived_fields`] is what restores them after any
/// edit to the tracks, the counts or the points' `w` values.
#[derive(Clone)]
pub struct PointSet {
    /// 3D points with colors, errors, and normals.
    pub points: Vec<Point3D>,
    /// Track observations (sorted by point_index, then image_index).
    pub tracks: Vec<TrackObservation>,
    /// Number of observations per 3D point.
    pub observation_counts: Vec<u32>,
    /// The observation-source-specific columns (per-observation feature index or
    /// keypoint, per-image hashes), selected by variant. The feature→point maps
    /// below are meaningful only for [`ObservationSource::SiftFiles`].
    ///
    /// The per-image hash vectors this carries are the one thing on the point
    /// side that is measured per image; they travel with the variant that
    /// selects them rather than splitting a second discriminator off into the
    /// image table.
    pub observations: ObservationSource,
    /// Optional per-point oriented-patch frame (parallel to `points`), persisted
    /// in `points3d/` (version 3+). `patch_u_halfvec_xyz` and
    /// `patch_v_halfvec_xyz` are the in-plane half-extent vectors (both present
    /// or both `None`); a patch's center is its point's position and its normal
    /// is the point's `normal`. See [`crate::patch::PatchCloud`].
    pub patch_u_halfvec_xyz: Option<Array2<f32>>,
    pub patch_v_halfvec_xyz: Option<Array2<f32>>,
    /// Optional `(P, R, R, 4)` per-point RGBA patch bitmaps; the alpha channel
    /// holds a per-pixel confidence.
    ///
    /// Behind an [`Arc`] because it is one of the two columns that dominate a
    /// reconstruction's memory, so two reconstructions that agree on their
    /// bitmaps share one copy. Nothing writes through it: a producer that
    /// changes the bitmaps builds a new array and wraps it.
    pub patch_bitmaps_y_x_rgba: Option<Arc<Array4<u8>>>,
    /// Whether this reconstruction carries per-point normals. When `false`, each
    /// point's inline `normal` is left zero and the columnar `normals_xyz` array
    /// is neither built nor written. `true` for everything loaded from versions 1
    /// and 2.
    pub has_normals: bool,
    /// Optional per-point confidence in each point's `normal` (parallel to
    /// `points`), persisted as `points3d/normal_confidence` (version 5+): `0`
    /// means the normal carries no data-derived support (a placeholder), `255`
    /// means fully data-derived, and intermediate values are a reserved graded
    /// scale. `None` means the reconstruction carries no confidence information
    /// at all — which is *not* the same as "all confident". It rides along
    /// untouched: nothing here synthesises or updates it when normals change.
    pub normal_confidence: Option<Vec<u8>>,
    /// Optional per-point solve constraints (parallel to `points`), persisted as
    /// the `points3d/point_constraints`, `points3d/constraint_distances` and
    /// `points3d/constraint_reference_images` triple
    /// (version 7+). `None` is every point free, which is what a file below
    /// version 7 carries and what the writer emits again when nothing is
    /// constrained.
    ///
    /// Nothing in this crate reads it to decide anything: it states what a
    /// bundle adjustment is to own of each point, and the adjustment's caller
    /// builds `PointConstraints` from it. Every pass that drops or reorders
    /// points selects its rows in lockstep with `points`, and every pass that
    /// drops or reindexes images moves the references with
    /// [`PointConstraintColumns::remap_images`].
    pub point_constraints: Option<PointConstraintColumns>,
    /// Optional per-observation confidence in that observation's **photometric
    /// sharpness relative to its track's consensus** (parallel to `tracks`),
    /// persisted as `tracks/observation_confidence` (version 6+): `0` means no
    /// data-derived support — nothing measured this observation — and `1..=255`
    /// is a measured scale running from maximally soft to fully sharp. `None`
    /// means the reconstruction carries no such information at all, which is
    /// *not* the same as "every observation is sharp".
    ///
    /// It is **metadata**: nothing in this crate reads it to decide anything. It
    /// rides along untouched, and every pass that drops or reorders observations
    /// selects its rows in lockstep with `tracks`.
    pub observation_confidence: Option<Vec<u8>>,

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
    /// [`Self::rebuild_derived_fields`] and by the in-place `w`-mutators
    /// (`classify_points_at_infinity` / `materialize_points_at_infinity`), since
    /// the count depends on point `w`-values rather than the track structure the
    /// other derived fields track.
    pub infinity_point_count: usize,
}

impl PointSet {
    /// Number of 3D points.
    pub fn point_count(&self) -> usize {
        self.points.len()
    }

    /// Number of track observations.
    pub fn observation_count(&self) -> usize {
        self.tracks.len()
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

    /// Per-observation sub-pixel keypoints `(M, 2)`, or `None` when this
    /// reconstruction carries none inline.
    ///
    /// Always present for `embedded_patches`, where the inline column *is* the
    /// observation coordinate; present for `sift_files` only when the file
    /// carries the optional inline copy.
    pub fn keypoints_xy(&self) -> Option<&Array2<f32>> {
        match &self.observations {
            ObservationSource::EmbeddedPatches { keypoints_xy, .. } => Some(keypoints_xy),
            ObservationSource::SiftFiles { keypoints_xy, .. } => keypoints_xy.as_ref(),
        }
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

    /// Return the observations for a given 3D point. O(1) lookup.
    pub fn observations_for_point(&self, point_idx: usize) -> &[TrackObservation] {
        let start = self.observation_offsets[point_idx];
        let end = self.observation_offsets[point_idx + 1];
        &self.tracks[start..end]
    }

    /// The observation row of the `(image, point, feature)` triple, or `None`
    /// when that image does not observe that point through that feature.
    ///
    /// A row index is what the per-observation columns (`keypoints_xy`,
    /// `observation_confidence`) are addressed by, while
    /// `image_feature_to_point` is keyed by feature; this walks the point's
    /// short observation run to cross the two.
    pub fn observation_row(
        &self,
        image_index: usize,
        point_index: u32,
        feature_index: u32,
    ) -> Option<usize> {
        let feature_indexes = self.feature_indexes()?;
        let start = self.observation_offsets[point_index as usize];
        self.observations_for_point(point_index as usize)
            .iter()
            .enumerate()
            .find(|(k, obs)| {
                obs.image_index as usize == image_index
                    && feature_indexes[start + k] == feature_index
            })
            .map(|(k, _)| start + k)
    }

    /// Return the image indices that observe a given 3D point.
    pub fn track_image_indices(&self, point_idx: usize) -> Vec<usize> {
        self.observations_for_point(point_idx)
            .iter()
            .map(|obs| obs.image_index as usize)
            .collect()
    }

    /// Rebuild derived fields (observation offsets, feature→point maps, and the
    /// `infinity_point_count` cache) from the current `tracks`,
    /// `observation_counts` and `points`, over `image_count` images.
    ///
    /// Call this after mutating tracks, observation counts, or point
    /// `w`-values externally. The two per-image maps are sized from
    /// `image_count`, which is the one thing the point side cannot see for
    /// itself; [`SfmrReconstruction::rebuild_derived_fields`] supplies it from
    /// the image table.
    ///
    /// [`SfmrReconstruction::rebuild_derived_fields`]: super::SfmrReconstruction::rebuild_derived_fields
    pub fn rebuild_derived_fields(&mut self, image_count: usize) {
        self.observation_offsets = compute_observation_offsets(&self.observation_counts);

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

    /// Check that the observation-source columns are parallel to the structures
    /// they annotate: per-observation columns (`feature_indexes` / `keypoints_xy`)
    /// must match the track count, and per-image columns (the hashes) must match
    /// `image_count`. Returns an error message describing the first mismatch.
    ///
    /// `from_sfmr_data` builds these in lockstep, but the in-memory editors
    /// (notably `clone_with_changes`, which can replace tracks and columns
    /// independently) can leave them out of step; this is the guard those paths
    /// run before handing back a reconstruction.
    pub fn validate_observation_columns(&self, image_count: usize) -> Result<(), String> {
        let n_obs = self.tracks.len();
        let n_img = image_count;
        // Mode-independent: an observation's confidence rates the observation,
        // not whichever column happens to back it.
        if let Some(confidence) = &self.observation_confidence {
            if confidence.len() != n_obs {
                return Err(format!(
                    "observation_confidence length ({}) must match observation count ({n_obs})",
                    confidence.len()
                ));
            }
        }
        match &self.observations {
            ObservationSource::SiftFiles {
                feature_indexes,
                keypoints_xy,
                feature_tool_hashes,
                sift_content_hashes,
            } => {
                if feature_indexes.len() != n_obs {
                    return Err(format!(
                        "feature_indexes length ({}) must match observation count ({n_obs})",
                        feature_indexes.len()
                    ));
                }
                if let Some(keypoints_xy) = keypoints_xy {
                    if keypoints_xy.nrows() != n_obs {
                        return Err(format!(
                            "keypoints_xy row count ({}) must match observation count ({n_obs})",
                            keypoints_xy.nrows()
                        ));
                    }
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
}
