// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The point side of a reconstruction: the 3D points, the tracks that observe
//! them, and everything measured per point or per observation.

use std::collections::HashMap;
use std::sync::Arc;

use ndarray::{Array2, Array4};

use sfmtool_sfmr_format::{
    FEATURE_SOURCE_EMBEDDED_PATCHES, FEATURE_SOURCE_SIFT_FILES, NO_REFERENCE_OBSERVATION,
};

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
    /// Whether `patch_bitmaps_y_x_rgba` was rendered for display rather than
    /// read or computed as part of the reconstruction.
    ///
    /// The viewer renders the column of a file that carries none and sets this,
    /// so the bench, the edits and the panels that read the column see it as
    /// they would a file's own. A column marked so is **left out of what
    /// [`SfmrReconstruction::to_sfmr_data`](super::SfmrReconstruction::to_sfmr_data)
    /// emits**, and so out of every save and every content hash: the value
    /// keeps the identity of the file it was read from, and a save writes the
    /// columns that file had. Every pass that selects or reorders the column's
    /// rows carries the mark with them; a producer that builds a new column
    /// clears it.
    pub patch_bitmaps_for_display: bool,
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
    /// Optional per-observation confidence in how well that observation agrees
    /// with its point's appearance (parallel to `tracks`), persisted as
    /// `tracks/observation_confidence` (version 6+): its blur-matched ZNCC against the
    /// point's stored patch bitmap, as `observation_confidence_byte` writes
    /// it. `0` means no data-derived support — nothing measured this
    /// observation — and `1..=255` is a measured score, `255` for the reference
    /// observation. `None` means the reconstruction carries no such information
    /// at all, which is *not* the same as "every observation agrees".
    ///
    /// It is **metadata**: nothing in this crate reads it to decide anything. It
    /// rides along untouched, and every pass that drops or reorders observations
    /// selects its rows in lockstep with `tracks`.
    pub observation_confidence: Option<Vec<u8>>,
    /// Optional per-observation readings on each observation's own `R×R`
    /// render (parallel to `tracks`): its self-similarity ellipse, the angle,
    /// tilt and zoom of that render, and its plain and blur-matched scores
    /// against the point's stored bitmap, persisted as the eight `tracks/`
    /// columns flagged by `has_observation_readings`. `None` where the
    /// reconstruction carries none.
    ///
    /// A row is a record of the render it names, not a claim about the
    /// current render. Every pass that drops or reorders observations selects
    /// its rows in lockstep with `tracks`, and a pass that moves geometry
    /// carries them unchanged; a writer that renders an observation's tile
    /// writes its row ([`Self::write_observation_readings`]).
    pub observation_readings: Option<super::ObservationReadings>,
    /// Per point (parallel to `points`), the index of its **reference
    /// observation** within its own track, `0` to `observation_counts[i] - 1`:
    /// the observation the point's patch bitmap is, or is to be, rendered
    /// from. With [`Self::patch_bitmaps_y_x_rgba`], the bitmap is that
    /// observation's `R×R` render; without it, the column stays, and a later
    /// render renders the point from it
    /// ([`render_patch_cloud_bitmaps`](crate::patch::stored_bitmap::render_patch_cloud_bitmaps)).
    /// [`NO_REFERENCE_OBSERVATION`] (`-1`) where the point has no reference
    /// observation in its track: a bitmap beside it is not the render of one
    /// of its observations (a fused mean, or the render of an observation an
    /// edit has since removed, which keeps the bitmap), and a later render
    /// runs the reference-view rule for the point. Persisted as
    /// `tracks/reference_observations` (version 12+).
    ///
    /// With a column rendered for display ([`Self::patch_bitmaps_for_display`]),
    /// a point the file stored at `-1` holds the observation the display render
    /// picked, so the bench and Track View mark the row its display bitmap is
    /// the tile of; [`Self::display_only_references`] marks those rows, and a
    /// save writes them as `-1` again ([`Self::saved_reference_observations`]),
    /// except for a point a bench commit has written, which holds the
    /// reference the bench held as its own.
    ///
    /// `Some` exactly when the patch frame is: a file below version 12 with
    /// patch frames loads with every row `-1`, and
    /// [`to_sfmr_data`](super::SfmrReconstruction::to_sfmr_data) writes `-1`
    /// rows for a value that has a frame and no column.
    ///
    /// Every pass that drops or reorders points selects its rows in lockstep
    /// with `points`. A pass that drops or reorders a point's observations
    /// moves the index with the reference observation
    /// ([`Self::select_reference_observations`]), `-1` where it drops it. A pass
    /// that renders a point's bitmap from another observation writes that
    /// observation's index.
    ///
    /// [`NO_REFERENCE_OBSERVATION`]: sfmtool_sfmr_format::NO_REFERENCE_OBSERVATION
    pub reference_observations: Option<Vec<i32>>,
    /// Per point (parallel to `points`), whether its entry in
    /// [`Self::reference_observations`] is a pick only the display render made
    /// -- the file stored `-1` for it, and the viewer rendered its display
    /// bitmap from the observation the reference-view rule picked. `Some` only
    /// beside a column marked [`Self::patch_bitmaps_for_display`].
    ///
    /// A display pick is shown, never saved, as the display bitmaps are not:
    /// [`Self::saved_reference_observations`] writes `-1` for a marked row.
    /// Every pass that drops or reorders points selects its rows in lockstep
    /// with `points` ([`Self::select_display_only_references`]); a point an
    /// edit rewrites through a whole record keeps its mark with the record,
    /// and a point an edit builds fresh, such as a bench commit, is unmarked,
    /// since its reference is the edit's own; a pass that drops or replaces the
    /// bitmap column clears it.
    pub display_only_references: Option<Vec<bool>>,

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

    /// The reference observations of a point set built from this one by
    /// selecting rows: `point_rows[q]` is the point of this set that new point
    /// `q` is, and `observation_rows` the observation rows of this set, in the
    /// new set's order, that the new tracks are. `None` where this set carries
    /// no column.
    ///
    /// Each new point's index is where its reference observation landed in its
    /// new run, `-1` where that observation was not selected or the point had
    /// no reference. An observation counts as the new point's when this set
    /// gives it to the point the new point is; the new run starts at the first
    /// such observation, so `observation_rows` must keep each point's
    /// observations together, as the tracks are.
    ///
    /// ```
    /// # use sfmtool_core::reconstruction::PointSet;
    /// # fn run(set: &PointSet) {
    /// // Keep every point, and drop each point's first observation.
    /// let points: Vec<usize> = (0..set.point_count()).collect();
    /// let rows: Vec<usize> = (0..set.point_count())
    ///     .flat_map(|p| set.observation_offsets[p] + 1..set.observation_offsets[p + 1])
    ///     .collect();
    /// let references = set.select_reference_observations(&points, &rows);
    /// # let _ = references;
    /// # }
    /// ```
    pub fn select_reference_observations(
        &self,
        point_rows: &[usize],
        observation_rows: &[usize],
    ) -> Option<Vec<i32>> {
        let references = self.reference_observations.as_ref()?;
        let mut new_point_of = vec![usize::MAX; self.points.len()];
        for (q, &p) in point_rows.iter().enumerate() {
            if new_point_of[p] == usize::MAX {
                new_point_of[p] = q;
            }
        }
        // Where each old observation row went, and where each new run starts.
        let mut new_row_of = vec![usize::MAX; self.tracks.len()];
        let mut run_start = vec![usize::MAX; point_rows.len()];
        for (k, &row) in observation_rows.iter().enumerate() {
            new_row_of[row] = k;
            let q = new_point_of[self.tracks[row].point_index as usize];
            if q != usize::MAX && run_start[q] == usize::MAX {
                run_start[q] = k;
            }
        }
        Some(
            point_rows
                .iter()
                .enumerate()
                .map(|(q, &p)| {
                    let r = references[p];
                    if r < 0 {
                        return NO_REFERENCE_OBSERVATION;
                    }
                    let row = self.observation_offsets[p] + r as usize;
                    match new_row_of.get(row) {
                        Some(&k) if k != usize::MAX && k >= run_start[q] => {
                            (k - run_start[q]) as i32
                        }
                        _ => NO_REFERENCE_OBSERVATION,
                    }
                })
                .collect(),
        )
    }

    /// [`Self::display_only_references`]'s rows for `point_rows`, the points a
    /// pass keeps, in their new order.
    pub fn select_display_only_references(&self, point_rows: &[usize]) -> Option<Vec<bool>> {
        let marks = self.display_only_references.as_ref()?;
        Some(point_rows.iter().map(|&p| marks[p]).collect())
    }

    /// Write the readings `rows` names, `(observation, reading)`, taken under
    /// `options`, into [`Self::observation_readings`].
    ///
    /// Where the reconstruction carries no readings, the column is created and
    /// every other row is [`ObservationReading::NOT_MEASURED`](super::ObservationReading::NOT_MEASURED).
    /// Where it carries readings taken under other options, every other row is
    /// cleared the same way, since the column records one set of options.
    ///
    /// # Panics
    ///
    /// Panics if an observation index is out of range.
    pub fn write_observation_readings(
        &mut self,
        rows: impl IntoIterator<Item = (usize, super::ObservationReading)>,
        options: super::ObservationReadingOptions,
    ) {
        let count = self.tracks.len();
        let readings = self
            .observation_readings
            .get_or_insert_with(|| super::ObservationReadings::not_measured(count, options));
        if readings.options != options || readings.rows.len() != count {
            *readings = super::ObservationReadings::not_measured(count, options);
        }
        for (j, row) in rows {
            readings.rows[j] = row;
        }
    }

    /// Clear the scores of point `point`'s observations to `NaN`, keeping the
    /// rest of each row: what a writer that changes the point's reference
    /// observation, and does not render its observations again, keeps.
    pub fn clear_observation_scores(&mut self, point: usize) {
        let (Some(readings), Some(range)) = (
            self.observation_readings.as_mut(),
            self.observation_offsets
                .get(point)
                .zip(self.observation_offsets.get(point + 1))
                .map(|(&a, &b)| a..b),
        ) else {
            return;
        };
        for row in &mut readings.rows[range] {
            *row = row.without_scores();
        }
    }

    /// [`Self::reference_observations`] as a save writes it: `-1` for each
    /// row [`Self::display_only_references`] marks, since a pick only the
    /// display render made is not the reconstruction's own.
    pub fn saved_reference_observations(&self) -> Option<Vec<i32>> {
        let mut references = self.reference_observations.clone()?;
        if let Some(marks) = &self.display_only_references {
            for (reference, &mark) in references.iter_mut().zip(marks) {
                if mark {
                    *reference = NO_REFERENCE_OBSERVATION;
                }
            }
        }
        Some(references)
    }

    /// Drop the patch bitmap column, keeping the reference observations,
    /// except a pick only the display render made
    /// ([`Self::display_only_references`]), which goes back to `-1`.
    pub fn drop_patch_bitmaps(&mut self) {
        self.reference_observations = self.saved_reference_observations();
        self.display_only_references = None;
        self.patch_bitmaps_y_x_rgba = None;
        self.patch_bitmaps_for_display = false;
    }

    /// The observation row of point `point`'s reference observation, or `None`
    /// where the set carries no column or the point has no reference.
    pub fn reference_observation_row(&self, point: usize) -> Option<usize> {
        let r = *self.reference_observations.as_ref()?.get(point)?;
        (r >= 0).then(|| self.observation_offsets[point] + r as usize)
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
        if let Some(readings) = &self.observation_readings {
            if readings.rows.len() != n_obs {
                return Err(format!(
                    "observation_readings length ({}) must match observation count ({n_obs})",
                    readings.rows.len()
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

/// An observation's score on the `tracks/observation_confidence` byte scale:
/// a measured ZNCC `z` is `round(255 · clamp(z, 0, 1))`, raised to at least `1`
/// so that it never reads as the `0` that means unmeasured; a non-finite score
/// (NaN, no measurement) is that `0`. Every writer of the column calls this, so
/// the bench commit and Add Image to Tracks store a score the same way.
pub(crate) fn observation_confidence_byte(zncc: f64) -> u8 {
    if zncc.is_finite() {
        ((zncc.clamp(0.0, 1.0) * f64::from(u8::MAX)).round() as u8).max(1)
    } else {
        0
    }
}
