// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Geometry-guided candidate search for a track-stage bench item.
//!
//! This is the single-point form of the view expansion used by
//! `sfm embed-patches`: project the track's surfel into every camera that can
//! see it, vet each projected appearance against a reference fused from the
//! selected observation and the track's accepted observations, and append the
//! admitted new images as candidates. Existing observations are reports, never
//! mutation targets.

use std::collections::BTreeSet;

use crate::patch::normal_refine::ProjectedImage;
use crate::patch::view_selection::{projected_patch_frame, select_patch_views, ViewSelectParams};
use crate::progress::{Cancelled, Progress};

use super::search::Found;
use super::steps::{add_observation, ObservationSeed};
use super::track::{ClusterPayload, EditableTrack, Provenance, Stage, StageKind, Verdict};

/// Tunables for [`search_geometry`].
#[derive(Debug, Clone, Default)]
pub struct GeometrySearchOptions {
    /// The patch-view selector's photometric and rendering tunables.
    ///
    /// `min_relative_zncc` is replaced by the editable track's threshold when
    /// the search runs. That bar belongs to the track, where the panel and
    /// threshold step expose it, rather than to one invocation.
    pub selection: ViewSelectParams,
}

/// Why a geometry search was refused.
#[derive(Debug, Clone, PartialEq)]
pub enum GeometrySearchError {
    /// The observation index is past the end of the track.
    NoSuchObservation {
        /// The index named.
        observation: usize,
        /// How many observations the track holds.
        observation_count: usize,
    },
    /// Geometry search is a track-stage operation.
    WrongStage {
        /// The stage the track is in.
        is: StageKind,
    },
    /// The track stage carries no surfel to project.
    NoFrame,
    /// The searched observation has no pixel at which to anchor its reference
    /// appearance.
    NoPlace {
        /// The observation named.
        observation: usize,
    },
    /// One of the reference observations names a view the caller did not
    /// supply.
    NoSuchImage {
        /// The image index named.
        image: u32,
        /// How many views were supplied.
        view_count: usize,
    },
    /// The caller asked the search to stop.
    Cancelled,
}

impl std::fmt::Display for GeometrySearchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoSuchObservation {
                observation,
                observation_count,
            } => write!(
                f,
                "observation {observation} is past the {observation_count} observations of this track"
            ),
            Self::WrongStage { is } => write!(
                f,
                "geometry search needs the track stage, and this track is a {is}"
            ),
            Self::NoFrame => write!(f, "this track has no surfel yet; fit it first"),
            Self::NoPlace { observation } => write!(
                f,
                "nothing says where observation {observation} sits in its photograph, so there is no reference appearance"
            ),
            Self::NoSuchImage { image, view_count } => write!(
                f,
                "image {image} is past the {view_count} decoded views supplied to the geometry search"
            ),
            Self::Cancelled => write!(f, "the geometry search was cancelled"),
        }
    }
}

impl std::error::Error for GeometrySearchError {}

impl From<Cancelled> for GeometrySearchError {
    fn from(_: Cancelled) -> Self {
        Self::Cancelled
    }
}

/// One image admitted by the geometric and photometric search.
#[derive(Debug, Clone, PartialEq)]
pub struct GeometryMatch {
    /// The image, as an index into the caller's view table.
    pub image: u32,
    /// Windowed ZNCC to the reference appearance.
    pub zncc: f64,
    /// Where the surfel's centre projects in the image.
    pub pixel: [f64; 2],
    /// What the search did about it.
    pub found: Found,
}

/// What one geometry search did.
#[derive(Debug, Clone, PartialEq)]
pub struct GeometrySearchReport {
    /// The observation whose appearance was explicitly selected as a reference.
    pub observation: usize,
    /// How many observations the track held when the search ran.
    pub observation_count: usize,
    /// The selected observation's image.
    pub image: u32,
    /// How many distinct selected/`in` images formed the reference basis.
    pub reference_views: usize,
    /// The reference basis's self-agreement; `NaN` when it could not be built.
    pub self_agreement: f64,
    /// Admitted non-reference images, in ascending image order.
    pub matches: Vec<GeometryMatch>,
}

impl GeometrySearchReport {
    /// How many candidate observations were appended.
    pub fn added(&self) -> usize {
        self.matches
            .iter()
            .filter(|m| matches!(m.found, Found::Added { .. }))
            .count()
    }

    /// How many admitted images the track already named.
    pub fn already_in_track(&self) -> usize {
        self.matches
            .iter()
            .filter(|m| matches!(m.found, Found::AlreadyInTrack { .. }))
            .count()
    }
}

impl std::fmt::Display for GeometrySearchReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Geometry search from observation {} of {}: ",
            self.observation, self.observation_count
        )?;
        if self.matches.is_empty() {
            return write!(f, "no new image passed the geometric and photometric gates");
        }
        write!(
            f,
            "{} images matched, {} candidates added",
            self.matches.len(),
            self.added()
        )?;
        let held = self.already_in_track();
        if held > 0 {
            write!(f, ", {held} already in the track")?;
        }
        Ok(())
    }
}

/// Project a track-stage surfel through all supplied views and append admitted
/// images as candidate observations.
///
/// `observation` chooses the source appearance: its image is first in the
/// reference basis even when its verdict is `candidate` or `out`. The other
/// basis images are the track's `in` observations, in observation order. Each
/// basis render is anchored at its observation's own pixel. This makes the row
/// gesture meaningful while retaining the robust multi-view reference used by
/// patch-view selection. Duplicate basis images are removed first-seen, so the
/// selected observation wins.
///
/// Finite surfels and direction surfels (`w == 0`) use
/// [`select_patch_views`]'s existing projection, front-facing,
/// cheirality, support, self-agreement and relative-ZNCC gates. Every admitted
/// view not already named by the track is appended at the surfel centre's
/// projection with the projected patch frame converted to the cluster seed
/// convention. Existing observations, verdicts and measurements are never
/// moved or overwritten.
///
/// Progress reports `build reference`, `score views`, and `add candidates`,
/// and cancellation returns no partially-grown track.
pub fn search_geometry(
    track: &EditableTrack,
    observation: usize,
    views: &[ProjectedImage<'_>],
    options: &GeometrySearchOptions,
    progress: &Progress<'_>,
) -> Result<(EditableTrack, GeometrySearchReport), GeometrySearchError> {
    let selected =
        track
            .observations
            .get(observation)
            .ok_or(GeometrySearchError::NoSuchObservation {
                observation,
                observation_count: track.observations.len(),
            })?;
    let payload = match &track.stage {
        Stage::Track(payload) => payload,
        Stage::Cluster(_) => {
            return Err(GeometrySearchError::WrongStage {
                is: StageKind::Cluster,
            })
        }
    };
    let frame = payload.frame.as_ref().ok_or(GeometrySearchError::NoFrame)?;
    selected
        .site()
        .ok_or(GeometrySearchError::NoPlace { observation })?;

    // The selected row is the explicit reference, then the accepted sightings
    // supply the rest of the robust basis. First-seen wins for two observations
    // in one image, matching view selection's own deduplication rule.
    let mut reference_rows = vec![observation];
    reference_rows.extend(
        track
            .observations
            .iter()
            .enumerate()
            .filter_map(|(i, row)| (row.verdict == Verdict::In && i != observation).then_some(i)),
    );
    let mut seen = BTreeSet::new();
    reference_rows.retain(|&i| seen.insert(track.observations[i].image));

    let mut reference_views = Vec::with_capacity(reference_rows.len());
    let mut reference_pixels = Vec::with_capacity(reference_rows.len());
    for &row_index in &reference_rows {
        let row = &track.observations[row_index];
        if row.image as usize >= views.len() {
            return Err(GeometrySearchError::NoSuchImage {
                image: row.image,
                view_count: views.len(),
            });
        }
        let pixel = row.site().ok_or(GeometrySearchError::NoPlace {
            observation: row_index,
        })?;
        reference_views.push(row.image);
        reference_pixels.push(Some(pixel));
    }

    let [select_progress, add_progress] = progress.split([0.95, 0.05]);
    let params = ViewSelectParams {
        min_relative_zncc: track.thresholds.min_relative_zncc,
        ..options.selection.clone()
    };
    let selection = select_patch_views(
        frame,
        views,
        &reference_views,
        Some(&reference_pixels),
        &params,
        &select_progress,
    )?;

    // All images already present are protected, regardless of verdict. The
    // selector may admit one that was not part of the reference basis; it is
    // reported and left byte-for-byte alone.
    let held: BTreeSet<u32> = track.observations.iter().map(|row| row.image).collect();
    let mut next = track.clone();
    let mut matches = Vec::new();
    {
        let candidate_progress = add_progress.phase("add candidates");
        let extras = selection.admitted.len() - selection.track_view_count;
        for (done, (&image, &zncc)) in selection.admitted[selection.track_view_count..]
            .iter()
            .zip(&selection.scores[selection.track_view_count..])
            .enumerate()
        {
            candidate_progress.check_cancel()?;
            let Some((pixel, patch_shape)) = projected_patch_frame(frame, &views[image as usize])
            else {
                continue;
            };
            let found = if held.contains(&image) {
                Found::AlreadyInTrack {
                    observation: next
                        .observations
                        .iter()
                        .position(|row| row.image == image)
                        .expect("the held image has an observation"),
                }
            } else {
                let radius = ClusterPayload::default().radius;
                // Patch-frame v points image-up, while a cluster/SIFT frame is
                // positive-chirality. Divide the projected half-vectors by the
                // cluster radius because a seed maps one keypoint-frame unit,
                // not the whole [-r, r] patch.
                let shape = [
                    [patch_shape[0][0] / radius, -patch_shape[0][1] / radius],
                    [patch_shape[1][0] / radius, -patch_shape[1][1] / radius],
                ];
                let seed = ObservationSeed {
                    image,
                    pixel,
                    shape: Some(shape),
                    provenance: Provenance::Sweep,
                };
                // The pixel came back from `ray_to_pixel`, so the only refusal
                // `add_observation` has left is one this cannot reach; an
                // image that somehow projected to a non-finite pixel is
                // skipped with the rest of its row rather than failing the
                // whole search, as the descriptor search skips its own.
                match add_observation(&next, &seed) {
                    Ok((grown, report)) => {
                        next = grown;
                        Found::Added {
                            observation: report.observation,
                        }
                    }
                    Err(_) => continue,
                }
            };
            matches.push(GeometryMatch {
                image,
                zncc,
                pixel,
                found,
            });
            candidate_progress.count((done + 1) as u64, Some(extras as u64), "candidate");
        }
    }

    Ok((
        next,
        GeometrySearchReport {
            observation,
            observation_count: track.observations.len(),
            image: selected.image,
            reference_views: reference_views.len(),
            self_agreement: selection.self_agreement,
            matches,
        },
    ))
}
