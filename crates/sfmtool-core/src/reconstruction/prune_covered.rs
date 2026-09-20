// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Retiring a reconstruction's coarse observations where a finer tracked one
//! already covers the same pixels.
//!
//! [`super::super::analysis::covered_by_finer`] is the rule, over flat rows and
//! knowing nothing about a reconstruction. This is that rule asked of a value: a
//! caller hands over the version it holds and gets back the version whose coarse
//! evidence has been handed over to the finer features that supersede it, the
//! map from its point indexes to that value's, and a report.
//!
//! Nothing is re-solved. No point moves, no camera moves, no lens moves; what
//! changes is which observations a point stands on, and which points are left
//! standing on too few.
//!
//! See `specs/core/reconstruction/prune-covered-observations.md` for the design.

use std::sync::Arc;

use ndarray::Axis;

use sfmtool_sfmr_format::{ContentHash, POINT_CONSTRAINT_HELD, POINT_CONSTRAINT_RANGED};

use crate::analysis::covered_by_finer::{
    covered_by_finer, CoveredByFinerError, CoveredCensus, CoveredOptions, CoveredRows,
};
use crate::progress::{Cancelled, Progress};
use crate::progress_info;
use crate::reconstruction::data::{
    ObservationSource, Point3D, PointSet, SfmrReconstruction, TrackObservation,
};
use crate::reconstruction::edited::{EditedReconstruction, PointMap, RowMap};

/// The rules one prune judges each observation by.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PruneCoveredOptions {
    /// What fraction of a row's projected patch radius its footprint is, which
    /// is the disk containment is asked within.
    ///
    /// The projected radius is the patch frame's own extent in that image, and
    /// a patch is embedded several feature sizes across: at the default patch
    /// size of 11 the frame spans 5.5 of them, so reading containment at the
    /// whole radius would ask it over a disk five times the feature. The
    /// half-extent a keypoint's support is stated at is 2.5 feature sizes
    /// ([`crate::patch::cloud::PatchExtent`]'s default), which on such a file is
    /// `2.5 / 5.5 = 0.4545` of the projected radius. The default rounds that up
    /// a little rather than reading the embedding size off the file, because a
    /// value carries no trustworthy statement of what it was embedded at and a
    /// guess at one would retire evidence on a number nobody measured.
    pub footprint_fraction: f64,
    /// How many times finer the covering row has to be. `2.0` is one octave.
    pub ratio: f64,
    /// A covering row whose projected radius is below this says nothing: a
    /// feature that projects to a fraction of a pixel is a collapsed
    /// measurement rather than finer evidence.
    pub min_fine_radius_px: f64,
    /// How many surviving observations a point needs to be kept.
    pub min_observations: usize,
}

impl Default for PruneCoveredOptions {
    fn default() -> Self {
        Self {
            footprint_fraction: 0.5,
            ratio: 2.0,
            min_fine_radius_px: 1.0,
            min_observations: 2,
        }
    }
}

/// Why a prune produced nothing. Every variant names what did not hold, because
/// the caller is a menu entry that has to say so in one sentence.
#[derive(Debug, Clone, PartialEq)]
pub enum PruneCoveredError {
    /// The points carry no patch frame, so no observation states a footprint.
    NoPatchFrames,
    /// The observations carry no pixel, so no footprint has a place to sit.
    NoKeypoints,
    /// No image of the reconstruction carries a usable pose, so nothing
    /// projects.
    NoPosedImages,
    /// The footprint fraction is not a usable multiple of the projected radius.
    BadFootprintFraction(f64),
    /// The rule refused the rows built for it.
    Rule(CoveredByFinerError),
    /// The caller asked the prune to stop, and it did, so there is no value to
    /// hand back.
    Cancelled,
}

impl std::fmt::Display for PruneCoveredError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoPatchFrames => write!(
                f,
                "pruning covered observations needs a patch frame per point to read each \
                 observation's footprint off, and this reconstruction carries none"
            ),
            Self::NoKeypoints => write!(
                f,
                "pruning covered observations needs a pixel per observation, and this \
                 reconstruction's observations are .sift feature indexes with no inline keypoints"
            ),
            Self::NoPosedImages => {
                write!(f, "no image of this reconstruction carries a pose")
            }
            Self::BadFootprintFraction(fraction) => write!(
                f,
                "the footprint fraction must be finite and above zero, and {fraction} is not"
            ),
            Self::Rule(e) => write!(f, "{e}"),
            Self::Cancelled => write!(
                f,
                "the prune was asked to stop before it had an answer, so nothing was written back"
            ),
        }
    }
}

impl std::error::Error for PruneCoveredError {}

impl From<Cancelled> for PruneCoveredError {
    fn from(_: Cancelled) -> Self {
        Self::Cancelled
    }
}

impl From<CoveredByFinerError> for PruneCoveredError {
    fn from(e: CoveredByFinerError) -> Self {
        match e {
            CoveredByFinerError::Cancelled => Self::Cancelled,
            other => Self::Rule(other),
        }
    }
}

/// One octave band of the projected radius, and what the prune did inside it.
///
/// The bands are anchored at the **largest** radius measured, so band 0 is the
/// coarsest octave and band `k` covers `[r_max / 2^(k + 1), r_max / 2^k)`. A
/// point's band is its widest observation's, which is the band a reader looking
/// for "where did this bite" means.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PruneCoveredBand {
    /// Which octave below the largest radius this is.
    pub band: usize,
    /// The band's lower bound in pixels, `r_max / 2^(band + 1)`.
    pub lower_px: f64,
    /// The band's upper bound in pixels, `r_max / 2^band`.
    pub upper_px: f64,
    /// Observations whose projected radius falls in the band.
    pub rows: usize,
    /// Of those, the ones the rule retired.
    pub rows_retired: usize,
    /// Points of the band the prune dropped.
    pub points_dropped: usize,
}

/// What one prune did.
#[derive(Debug, Clone, PartialEq)]
pub struct PruneCoveredReport {
    /// What the rule saw, over the rows it was handed.
    pub census: CoveredCensus,
    /// Observations whose patch frame does not project to a usable radius:
    /// behind the camera, degenerate, or of a point with no frame. Such a row
    /// is neither retired nor covers.
    pub degenerate_rows: usize,
    /// Observations of a point the value ranges or holds. They are never
    /// retired, and they still cover.
    pub protected_rows: usize,
    /// Of those, the ones a finer observation would otherwise have retired.
    pub protected_rows_spared: usize,
    /// Points before the prune.
    pub points_before: usize,
    /// Points after it.
    pub points_after: usize,
    /// Observations before the prune.
    pub observations_before: usize,
    /// Observations after it.
    pub observations_after: usize,
    /// Whether anything was retired. A prune that retires nothing hands the
    /// value back untouched and the caller pushes no version for it.
    pub changed: bool,
    /// Per octave band of the projected radius, coarsest first; empty when no
    /// row projected to a usable radius.
    pub bands: Vec<PruneCoveredBand>,
}

/// Retire every observation of `edited` that a finer tracked one covers, and
/// hand back the value that is left.
///
/// Per observation the rule reads two lengths off the same projection. The
/// **radius** is the mean of the two column norms of the point's patch frame
/// projected into the observing camera -- the size measure the Track View
/// reports and the one a `.sift` affine shape yields -- and the **footprint**
/// containment is asked within is [`PruneCoveredOptions::footprint_fraction`] of
/// it. An observation is retired where another observation in the same image, on
/// another point, sits inside that footprint with a radius at least
/// [`PruneCoveredOptions::ratio`] times smaller.
///
/// The rule is image-space throughout, so a point at infinity is read like any
/// other: its patch is tangent to the direction sphere and projects to a
/// footprint in every image that sees it.
///
/// **A constrained point is protected.** A point the value ranges or holds is a
/// statement somebody made by hand, so none of its observations is retired
/// however well covered; they still cover other points' observations, because
/// what protection refuses is the retirement and not the reading.
///
/// **A point left under [`PruneCoveredOptions::min_observations`] is dropped**,
/// and its surviving observations with it, so no caller is handed a track the
/// value would not keep.
///
/// The value that comes back is a whole new base with an empty overlay, and the
/// map is the one [`RowMap::by_scan`] reads off this call's input and output. A
/// prune that retires nothing hands `edited` back as it stands, with an empty
/// map and [`PruneCoveredReport::changed`] false.
///
/// `progress` names the stages (folding the overlay, measuring the footprints,
/// the rule, writing the value back) and is how the call is asked to stop. A
/// cancel is read between stages: a stopped call returns
/// [`PruneCoveredError::Cancelled`] and writes nothing. Pass `&Progress::none()`
/// to report nothing and never stop.
///
/// # Example
///
/// ```no_run
/// use sfmtool_core::progress::Progress;
/// use sfmtool_core::reconstruction::prune_covered::{
///     prune_covered_observations, PruneCoveredOptions,
/// };
/// # fn run(edited: &sfmtool_core::EditedReconstruction)
/// # -> Result<(), Box<dyn std::error::Error>> {
/// let (next, map, report) = prune_covered_observations(
///     edited,
///     &PruneCoveredOptions::default(),
///     &Progress::none(),
/// )?;
/// println!(
///     "{} of {} observations retired, {} points dropped",
///     report.census.rows_removed,
///     report.observations_before,
///     report.points_before - report.points_after
/// );
/// # let _ = (next, map);
/// # Ok(())
/// # }
/// ```
///
/// # Errors
///
/// [`PruneCoveredError`] states which precondition did not hold, or that the
/// call was cancelled.
pub fn prune_covered_observations(
    edited: &EditedReconstruction,
    options: &PruneCoveredOptions,
    progress: &Progress<'_>,
) -> Result<(EditedReconstruction, PointMap, PruneCoveredReport), PruneCoveredError> {
    progress.check_cancel()?;
    if !edited.has_patch_frames() {
        return Err(PruneCoveredError::NoPatchFrames);
    }
    if !edited.has_keypoints() {
        return Err(PruneCoveredError::NoKeypoints);
    }
    if edited.posed_lens_count() == 0 {
        return Err(PruneCoveredError::NoPosedImages);
    }
    if !(options.footprint_fraction.is_finite() && options.footprint_fraction > 0.0) {
        return Err(PruneCoveredError::BadFootprintFraction(
            options.footprint_fraction,
        ));
    }
    // Measuring the footprints is one projection per observation and the write
    // is one pass over every column; the rule between them is the enumeration,
    // which is a sort and the pairs it finds. An estimate, as every set of
    // weights is.
    let [p_fold, p_measure, p_rule, p_write] = progress.split([0.10, 0.35, 0.25, 0.30]);

    // Fold the overlay only when there is one: an empty overlay materialises to
    // its own base, and the measurement can read that directly.
    let folded = !(edited.deleted_points.is_empty() && edited.added.points.is_empty());
    let (materialised, mat_map) = if folded {
        let _phase = p_fold.phase("materialise");
        let (value, map) = edited.materialize();
        (Some(value), Some(PointMap::Rows(map)))
    } else {
        (None, None)
    };
    let source: &SfmrReconstruction = match materialised.as_ref() {
        Some(value) => value,
        None => &edited.base,
    };
    progress.check_cancel()?;

    let measured = {
        let mut phase = p_measure.phase("measure footprints");
        let measured = measure(source, options);
        crate::progress_note!(
            phase,
            "{} of {} rows project to a radius",
            measured.rows - measured.degenerate,
            measured.rows
        );
        measured
    };
    progress.check_cancel()?;

    let verdict = covered_by_finer(
        CoveredRows {
            image_of_row: &measured.image_of_row,
            owner_of_row: &measured.point_of_row,
            xy_px: &measured.xy_px,
            reach_px: &measured.reach_px,
            radius_px: &measured.radius_px,
            protected: Some(&measured.protected),
        },
        source.point_count(),
        &CoveredOptions {
            ratio: options.ratio,
            min_fine_radius_px: options.min_fine_radius_px,
            min_observations: options.min_observations,
        },
        &p_rule,
    )?;
    progress.check_cancel()?;

    let mut report = PruneCoveredReport {
        census: verdict.census,
        degenerate_rows: measured.degenerate,
        protected_rows: measured.protected.iter().filter(|&&p| p).count(),
        protected_rows_spared: verdict.census.rows_spared,
        points_before: source.point_count(),
        points_after: source.point_count(),
        observations_before: measured.rows,
        observations_after: measured.rows,
        changed: verdict.census.rows_removed > 0,
        bands: bands(&measured, &verdict.flagged, &verdict.keep_owner),
    };
    if !report.changed {
        progress_info!(progress, "nothing was covered, so nothing was retired");
        return Ok((edited.clone(), PointMap::Chain(Vec::new()), report));
    }

    let (next, map) = {
        let _phase = p_write.phase("write back");
        let pruned = prune_rows(source, &verdict.keep_row, &verdict.keep_owner);
        // The prune drops points and moves none, and it knows exactly which,
        // so it states the map rather than having it read back off the two
        // values: a value whose points are all seen by the same images is one
        // `RowMap::by_scan` cannot tell a dropped point from its neighbour on.
        let dropped: Vec<u32> = (0..source.point_count() as u32)
            .filter(|&p| !verdict.keep_owner[p as usize])
            .collect();
        let rows = RowMap::by_removal(source.point_count() as u32, &dropped);
        report.points_after = pruned.point_count();
        report.observations_after = pruned.point_set.observation_count();
        let mut steps = Vec::new();
        steps.extend(mat_map);
        steps.push(PointMap::Rows(rows));
        (
            EditedReconstruction::new(Arc::new(pruned)),
            PointMap::Chain(steps),
        )
    };
    progress_info!(
        progress,
        "{} of {} observations retired, {} of {} points dropped",
        report.census.rows_removed,
        report.observations_before,
        report.points_before - report.points_after,
        report.points_before
    );
    Ok((next, map, report))
}

/// The per-observation rows the rule reads, in the value's own row order.
struct Measured {
    rows: usize,
    image_of_row: Vec<i64>,
    point_of_row: Vec<i64>,
    xy_px: Vec<f64>,
    reach_px: Vec<f64>,
    radius_px: Vec<f64>,
    protected: Vec<bool>,
    degenerate: usize,
}

/// Project every point's patch frame into every image that observes it, and
/// read the two lengths the rule needs off each projection.
///
/// A row whose frame does not project -- no frame, behind the camera, a
/// degenerate or non-finite shape, or a radius at or below zero -- states `NaN`
/// for both lengths. That is the enumeration's "asks nothing" value on the
/// reach, and on the radius it fails every comparison, so such a row is neither
/// retired nor able to retire anything.
fn measure(source: &SfmrReconstruction, options: &PruneCoveredOptions) -> Measured {
    let n = source.point_set.tracks.len();
    let mut out = Measured {
        rows: n,
        image_of_row: Vec::with_capacity(n),
        point_of_row: Vec::with_capacity(n),
        xy_px: Vec::with_capacity(2 * n),
        reach_px: Vec::with_capacity(n),
        radius_px: Vec::with_capacity(n),
        protected: Vec::with_capacity(n),
        degenerate: 0,
    };
    let keypoints = source
        .keypoints_xy()
        .expect("the caller refused a value with no inline keypoints");
    for point in 0..source.point_count() {
        let start = source.point_set.observation_offsets[point];
        let held = source
            .point_set
            .point_constraints
            .as_ref()
            .is_some_and(|c| {
                let kind = c.point_constraints[point];
                kind == POINT_CONSTRAINT_HELD || kind == POINT_CONSTRAINT_RANGED
            });
        for (k, observation) in source.observations_for_point(point).iter().enumerate() {
            let row = start + k;
            let xy = [keypoints[[row, 0]], keypoints[[row, 1]]];
            let image = observation.image_index as usize;
            let radius = source
                .observation_affine_shape(point, image, xy)
                .map(|a| {
                    let col0 = f64::from(a[0][0] * a[0][0] + a[1][0] * a[1][0]).sqrt();
                    let col1 = f64::from(a[0][1] * a[0][1] + a[1][1] * a[1][1]).sqrt();
                    0.5 * (col0 + col1)
                })
                .filter(|r| r.is_finite() && *r > 0.0);
            if radius.is_none() {
                out.degenerate += 1;
            }
            let radius = radius.unwrap_or(f64::NAN);
            out.image_of_row.push(observation.image_index as i64);
            out.point_of_row.push(point as i64);
            out.xy_px.push(f64::from(xy[0]));
            out.xy_px.push(f64::from(xy[1]));
            out.reach_px.push(options.footprint_fraction * radius);
            out.radius_px.push(radius);
            out.protected.push(held);
        }
    }
    out
}

/// The octave-band table, anchored at the largest radius measured.
fn bands(measured: &Measured, flagged: &[bool], keep_owner: &[bool]) -> Vec<PruneCoveredBand> {
    let Some(r_max) = measured
        .radius_px
        .iter()
        .copied()
        .filter(|r| r.is_finite())
        .fold(None::<f64>, |best, r| Some(best.map_or(r, |b| b.max(r))))
    else {
        return Vec::new();
    };
    let band_of = |radius: f64| -> Option<usize> {
        radius
            .is_finite()
            .then(|| (r_max / radius).log2().floor().max(0.0) as usize)
    };

    // A point's band is its widest observation's, which is the lowest band
    // index any of its rows falls in.
    let mut point_band: Vec<Option<usize>> = vec![None; keep_owner.len()];
    let mut rows = Vec::new();
    let mut retired = Vec::new();
    for row in 0..measured.rows {
        let Some(band) = band_of(measured.radius_px[row]) else {
            continue;
        };
        if rows.len() <= band {
            rows.resize(band + 1, 0usize);
            retired.resize(band + 1, 0usize);
        }
        rows[band] += 1;
        retired[band] += usize::from(flagged[row]);
        let slot = &mut point_band[measured.point_of_row[row] as usize];
        *slot = Some(slot.map_or(band, |b| b.min(band)));
    }

    let mut dropped = vec![0usize; rows.len()];
    for (point, band) in point_band.iter().enumerate() {
        if let Some(band) = band {
            if !keep_owner[point] {
                dropped[*band] += 1;
            }
        }
    }

    (0..rows.len())
        .filter(|&band| rows[band] > 0)
        .map(|band| PruneCoveredBand {
            band,
            lower_px: r_max / 2f64.powi(band as i32 + 1),
            upper_px: r_max / 2f64.powi(band as i32),
            rows: rows[band],
            rows_retired: retired[band],
            points_dropped: dropped[band],
        })
        .collect()
}

/// `source` with the rows `keep_row` refuses and the points `keep_point`
/// refuses taken out, and the surviving points renumbered.
///
/// Every column but the tracks travels verbatim: a surviving point keeps its
/// position, its frame, its bitmap, its colour and its constraint, and only its
/// observation list is shorter.
fn prune_rows(
    source: &SfmrReconstruction,
    keep_row: &[bool],
    keep_point: &[bool],
) -> SfmrReconstruction {
    let set = &source.point_set;
    let keep_idx: Vec<usize> = (0..set.points.len()).filter(|&p| keep_point[p]).collect();
    let mut point_remap = vec![u32::MAX; set.points.len()];
    for (new, &old) in keep_idx.iter().enumerate() {
        point_remap[old] = new as u32;
    }
    let kept_obs: Vec<usize> = (0..set.tracks.len()).filter(|&row| keep_row[row]).collect();

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
        has_normals: set.has_normals,
        normal_confidence: set
            .normal_confidence
            .as_ref()
            .map(|c| keep_idx.iter().map(|&p| c[p]).collect()),
        // A constraint describes its own point and the images are untouched, so
        // the surviving rows travel verbatim.
        point_constraints: set.point_constraints.as_ref().map(|c| c.select(&keep_idx)),
        observation_confidence: set
            .observation_confidence
            .as_ref()
            .map(|c| kept_obs.iter().map(|&row| c[row]).collect()),
        observation_offsets: Vec::new(),
        image_feature_to_point: Vec::new(),
        max_track_feature_index: Vec::new(),
        infinity_point_count: 0,
    };
    point_set.rebuild_derived_fields(source.image_count());

    let mut metadata = source.metadata.clone();
    metadata.point_count = point_set.points.len() as u32;
    metadata.observation_count = point_set.tracks.len() as u32;
    metadata.infinity_point_count = point_set.infinity_point_count as u32;
    SfmrReconstruction {
        workspace_dir: source.workspace_dir.clone(),
        metadata,
        // The hashes on a value describe the file it came from, and this value
        // is not that file.
        content_hash: ContentHash::default(),
        // The prune touches no image, so the whole table is the source's,
        // thumbnails shared rather than copied.
        image_table: source.image_table.clone(),
        point_set,
    }
}

#[cfg(test)]
mod tests;
