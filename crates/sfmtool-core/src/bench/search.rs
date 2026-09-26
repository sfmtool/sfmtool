// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The descriptor search: which other photographs of the capture hold the patch
//! around one observation, and where in each of them it sits.
//!
//! `specs/core/bench/editable-track.md` is the design. This is the third way an
//! observation reaches a track, beside the pixel someone pointed at and the
//! point one was put on the bench from, and it is the only one that proposes
//! more than one candidate at a time.
//!
//! It is a [patch constellation
//! query](crate::features::kdforest::constellation_from_keypoints) and not a
//! lookup of one descriptor. A pixel someone pointed at is an extremum of
//! nothing, so a descriptor computed there matches nothing a detector produced
//! for the same surface elsewhere; what is stable is the **neighbourhood** of
//! detected keypoints around it, which another view of the same surface carries
//! under a locally affine warp. So the query asks the index about the keypoints
//! inside a radius of the observation, groups the hits by image, and keeps the
//! images whose hits agree on one warp. That warp is what makes the answer
//! useful: applied to the observation's own pixel and affine shape, it gives
//! each found image a seed position and shape, whether the observation being
//! searched from was a detected feature or a hand-placed pixel.
//!
//! The step reads no photograph. It reads the index, which is file I/O, so a
//! caller that must not block runs it where its other I/O runs.

use std::collections::BTreeSet;

use crate::features::kdforest::{
    constellation_from_keypoints, ConstellationParams, ImageKeypoints, KdfError, LazyKdForestU8,
};
use crate::progress::{Cancelled, Progress};

use super::steps::{add_observation, reference_shape, ObservationSeed};
use super::track::{EditableTrack, Provenance};

#[cfg(test)]
mod tests;

/// The radius the search takes when a caller names none, in source-image px.
///
/// Fifty keypoints is the constellation size to ask for
/// (`crate::features::kdforest::radius_for_feature_count`), and this is the
/// radius that rule gives a full-frame capture at that size -- a 2160 x 3840
/// frame with eight thousand keypoints in it. A caller that knows its frame and
/// its keypoint count computes its own with that rule instead, which is what the
/// viewer does; this is what a caller with neither is worth.
pub const DEFAULT_RADIUS_PX: f32 = 128.0;

/// What the search is allowed to do.
#[derive(Debug, Clone)]
pub struct SearchOptions {
    /// The constellation query's own tunables: how many neighbours each
    /// keypoint retrieves, the RANSAC budget and threshold, and the scale
    /// change a warp may claim.
    ///
    /// Its `min_inliers` is **not** read: the bar lives on [`Self::min_inliers`]
    /// below, which is written into the params the query runs with, so the
    /// number the report states a refusal against and the number the query
    /// fits to are one number in one place.
    pub constellation: ConstellationParams,
    /// The radius around the observation the constellation is taken from, in
    /// that image's own pixels.
    ///
    /// A patch, not a frame: the affine is the first-order approximation of the
    /// homography about the patch centre, and the term it drops grows with the
    /// patch, so a wide radius buys image recall and loses the accuracy of the
    /// warp that makes a found image worth seeding.
    pub radius_px: f32,
    /// Fewest correspondences an image's warp needs before it becomes a
    /// candidate.
    pub min_inliers: usize,
}

impl Default for SearchOptions {
    fn default() -> Self {
        Self {
            constellation: ConstellationParams::DEFAULT,
            radius_px: DEFAULT_RADIUS_PX,
            min_inliers: ConstellationParams::DEFAULT.min_inliers,
        }
    }
}

/// Why a search was refused. Each variant names what did not hold, because the
/// caller is a menu entry that has to say so in one sentence.
#[derive(Debug, Clone, PartialEq)]
pub enum SearchError {
    /// The observation index is past the end of the track.
    NoSuchObservation {
        /// The index named.
        observation: usize,
        /// How many observations the track holds.
        observation_count: usize,
    },
    /// Nothing says where the observation sits in its photograph, so there is
    /// no pixel to take a constellation around.
    NoPlace {
        /// The observation named.
        observation: usize,
    },
    /// The radius holds no keypoint the index knows, so there is no
    /// constellation to ask about.
    NoConstellation {
        /// The radius that held none, in source-image px.
        radius_px: f32,
        /// Keypoints the image has in total, so a reader can tell an empty
        /// `.sift` file from a radius that is too small.
        keypoint_count: usize,
    },
    /// The index refused, in its own words.
    Index(String),
    /// The caller asked the search to stop.
    Cancelled,
}

impl std::fmt::Display for SearchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SearchError::NoSuchObservation {
                observation,
                observation_count,
            } => write!(
                f,
                "observation {observation} is past the {observation_count} \
                 observations of this track"
            ),
            SearchError::NoPlace { observation } => write!(
                f,
                "nothing says where observation {observation} sits in its photograph, \
                 so there is no patch to search from"
            ),
            SearchError::NoConstellation {
                radius_px,
                keypoint_count,
            } => write!(
                f,
                "no indexed keypoint sits within {radius_px:.0} px of the observation, \
                 out of {keypoint_count} in the image"
            ),
            SearchError::Index(message) => write!(f, "the descriptor index refused: {message}"),
            SearchError::Cancelled => write!(f, "the search was cancelled"),
        }
    }
}

impl std::error::Error for SearchError {}

impl From<Cancelled> for SearchError {
    fn from(_: Cancelled) -> Self {
        SearchError::Cancelled
    }
}

impl From<KdfError> for SearchError {
    fn from(e: KdfError) -> Self {
        SearchError::Index(e.to_string())
    }
}

/// What became of one image the search found the patch in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Found {
    /// It became a candidate observation of the track.
    Added {
        /// The index the new observation took, which is the end of the list
        /// and is stable for the life of the track.
        observation: usize,
    },
    /// The track already has an observation in that image, whatever its
    /// verdict. A refusal is a decision the search does not overturn, and a
    /// candidate is one already on the table.
    AlreadyInTrack {
        /// The observation that holds the image.
        observation: usize,
    },
    /// The image the search was run from. The query drops its own image, so
    /// this is a guard rather than an outcome a corpus produces.
    OwnImage,
}

/// One image the patch was found in, and what the search did about it.
#[derive(Debug, Clone, PartialEq)]
pub struct SearchMatch {
    /// The image, as an index into the node's image table.
    pub image: u32,
    /// Correspondences that agreed with the warp.
    pub inliers: usize,
    /// Correspondences the image had before the fit.
    pub correspondences: usize,
    /// Row-major 2x3 affine taking the searched image's pixels to this image's:
    /// `x' = affine[0][0] * x + affine[0][1] * y + affine[0][2]`, and likewise
    /// `y'` from `affine[1]`.
    pub affine: [[f64; 3]; 2],
    /// Where the warp puts the observation's own pixel in this image.
    pub pixel: [f64; 2],
    /// What the search did about it.
    pub found: Found,
}

/// What one search did.
#[derive(Debug, Clone, PartialEq)]
pub struct SearchReport {
    /// The observation it was run from.
    pub observation: usize,
    /// How many observations the track held when it ran, so the sentence can
    /// say "observation 3 of 12" without a second lookup.
    pub observation_count: usize,
    /// The image that observation is in.
    pub image: u32,
    /// The pixel the constellation was taken around.
    pub center: [f64; 2],
    /// Keypoints in the constellation: inside the radius, and indexed.
    pub constellation: usize,
    /// The images the patch was found in, most inliers first, each with what
    /// the search did about it.
    pub matches: Vec<SearchMatch>,
}

impl SearchReport {
    /// How many candidates were added.
    pub fn added(&self) -> usize {
        self.matches
            .iter()
            .filter(|m| matches!(m.found, Found::Added { .. }))
            .count()
    }

    /// How many found images the track already had an observation in.
    pub fn already_in_track(&self) -> usize {
        self.matches
            .iter()
            .filter(|m| matches!(m.found, Found::AlreadyInTrack { .. }))
            .count()
    }
}

impl std::fmt::Display for SearchReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Searched from observation {} of {}: ",
            self.observation, self.observation_count
        )?;
        if self.matches.is_empty() {
            return write!(
                f,
                "no image matched the {} keypoints of it",
                self.constellation
            );
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

/// Find the images that hold the patch around one observation, and add each as
/// a candidate.
///
/// `keypoints` are the **searched image's** detected keypoints, which the
/// caller supplies: a window hands over what its own feature cache already
/// holds, and a script reads them from the `.sift` file with
/// [`ImageKeypoints::read`]. Nothing here opens one.
///
/// `forest`'s corpus **indexes this reconstruction's images, in the same
/// order**: a match names a corpus image index and the observation it becomes
/// names a node image index, and the search states them to be one number rather
/// than carrying a name table it has no reconstruction to check against. A
/// caller that opens an index checks that where it opens it.
///
/// The seed each candidate takes is the cluster stage's own
/// (`specs/core/bench/editable-track.md` § "The cluster stage's units"): the
/// warp applied to the searched observation's pixel, and its linear part
/// applied to the shape that observation carries -- its own refined or seeded
/// shape, else the cluster's reference's, else the identity, which is the same
/// order [`add_observation`] falls back through. So a candidate arrives at the
/// size and orientation the warp says the patch has in that image, and an
/// evaluation is what then scores it.
///
/// Works at both stages. At the cluster stage the seed is what the refinement
/// reads; at the track stage the candidate's pixel is also its keypoint, and
/// it is a row the next reading measures and the thresholds propose a verdict
/// for. The step sets no verdict and moves
/// no observation that was already on the track.
///
/// `progress` names one phase, `query index`: the corpus reads that resolve the
/// searched image's keypoints to feature IDs, the forest traversal, and the
/// per-image fit. Everything outside it is arithmetic over the track.
///
/// # Example
///
/// ```no_run
/// # use sfmtool_core::bench::{search_descriptors, SearchOptions};
/// # use sfmtool_core::features::kdforest::{ImageKeypoints, LazyKdForestU8};
/// # use sfmtool_core::progress::Progress;
/// # fn run(
/// #     track: &sfmtool_core::bench::EditableTrack,
/// #     keypoints: &ImageKeypoints,
/// #     forest: &LazyKdForestU8,
/// # ) -> Result<(), Box<dyn std::error::Error>> {
/// let (grown, report) = search_descriptors(
///     track,
///     0,
///     keypoints,
///     forest,
///     &SearchOptions::default(),
///     &Progress::none(),
/// )?;
/// println!("{report}");   // "Searched from observation 0 of 1: 7 images matched, …"
/// # let _ = grown;
/// # Ok(())
/// # }
/// ```
pub fn search_descriptors(
    track: &EditableTrack,
    observation: usize,
    keypoints: &ImageKeypoints,
    forest: &LazyKdForestU8,
    options: &SearchOptions,
    progress: &Progress<'_>,
) -> Result<(EditableTrack, SearchReport), SearchError> {
    let row = track
        .observations
        .get(observation)
        .ok_or(SearchError::NoSuchObservation {
            observation,
            observation_count: track.observations.len(),
        })?;
    let center =
        observation_pixel(track, observation).ok_or(SearchError::NoPlace { observation })?;
    let shape = observation_shape(track, observation);
    let image = row.image;

    // The bar is stated once, on the options, and written into the query so an
    // image the search would discard is never fitted in the first place.
    let params = ConstellationParams {
        min_inliers: options.min_inliers,
        ..options.constellation
    };
    // The two places a cancel can land: in front of the forest query, which is
    // one call and runs to its end once entered, and between the candidates it
    // found, each of which grows the track.
    progress.check_cancel()?;
    let found = {
        let _phase = progress.phase("query index");
        constellation_from_keypoints(
            forest,
            forest,
            keypoints,
            image,
            [center[0] as f32, center[1] as f32],
            options.radius_px,
            &params,
        )?
    };
    if found.feature_rows.is_empty() {
        return Err(SearchError::NoConstellation {
            radius_px: options.radius_px,
            keypoint_count: keypoints.len(),
        });
    }

    // Every image the track already names, whatever the verdict: a search does
    // not overturn a refusal and does not propose a second copy of a candidate.
    let held: BTreeSet<u32> = track.observations.iter().map(|o| o.image).collect();

    let mut next = track.clone();
    let mut matches = Vec::with_capacity(found.matches.len());
    for candidate in &found.matches {
        progress.check_cancel()?;
        let pixel = apply_affine(&candidate.affine, center);
        let outcome = if candidate.image_index == image {
            Found::OwnImage
        } else if held.contains(&candidate.image_index) {
            Found::AlreadyInTrack {
                observation: next
                    .observations
                    .iter()
                    .position(|o| o.image == candidate.image_index)
                    .expect("the image is held, so some observation names it"),
            }
        } else {
            let seed = ObservationSeed {
                image: candidate.image_index,
                pixel,
                shape: Some(apply_linear(&candidate.affine, shape)),
                provenance: Provenance::Search {
                    inliers: candidate.inliers as u32,
                },
            };
            // The seed is finite and the shape is the caller's to judge, so the
            // only refusal `add_observation` has left is one this cannot reach;
            // an image that somehow produced a non-finite warp is skipped with
            // the rest of its row rather than failing the whole search.
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
        matches.push(SearchMatch {
            image: candidate.image_index,
            inliers: candidate.inliers,
            correspondences: candidate.correspondences,
            affine: candidate.affine,
            pixel,
            found: outcome,
        });
    }

    Ok((
        next,
        SearchReport {
            observation,
            observation_count: track.observations.len(),
            image,
            center,
            constellation: found.feature_rows.len(),
            matches,
        },
    ))
}

/// Where one observation sits in its photograph: the track stage's keypoint,
/// else the cluster stage's refined position or the seed it started from.
///
/// The same order everything that draws an observation uses, so the pixel a
/// search runs from is the mark the person is looking at. There is no third
/// source: projecting the track's point would need the reconstruction, which
/// this step does not take, and an observation with neither a keypoint nor a
/// seed is refused rather than guessed at.
fn observation_pixel(track: &EditableTrack, observation: usize) -> Option<[f64; 2]> {
    let row = track.observations.get(observation)?;
    if let Some(keypoint) = row.track.as_ref().and_then(|m| m.keypoint) {
        return Some([f64::from(keypoint[0]), f64::from(keypoint[1])]);
    }
    Some(row.cluster.as_ref()?.best_position())
}

/// The keypoint-frame shape a candidate's own is warped from: the observation's
/// refined or seeded shape, else the cluster's reference's, else the identity.
///
/// The identity is one pixel to the keypoint-frame unit, which is what
/// [`add_observation`] falls back to for a track that has nothing to say about
/// its own scale; stating the same order here keeps a searched candidate and a
/// hand-placed one at one size.
fn observation_shape(track: &EditableTrack, observation: usize) -> [[f64; 2]; 2] {
    track
        .observations
        .get(observation)
        .and_then(|row| row.cluster.as_ref())
        .map(|m| m.shape.unwrap_or(m.seed_shape))
        .or_else(|| reference_shape(track))
        .unwrap_or([[1.0, 0.0], [0.0, 1.0]])
}

/// `affine` applied to a point.
fn apply_affine(affine: &[[f64; 3]; 2], p: [f64; 2]) -> [f64; 2] {
    [
        affine[0][0] * p[0] + affine[0][1] * p[1] + affine[0][2],
        affine[1][0] * p[0] + affine[1][1] * p[1] + affine[1][2],
    ]
}

/// The linear part of `affine` applied to a keypoint-frame shape.
///
/// A shape's columns are the images of the frame's axes in pixels, so the warp
/// acts on them from the left: `S' = A S`, where `A` is the warp's 2x2 part.
/// The translation plays no part, which is what makes a shape a scale rather
/// than a place.
fn apply_linear(affine: &[[f64; 3]; 2], shape: [[f64; 2]; 2]) -> [[f64; 2]; 2] {
    let a = [[affine[0][0], affine[0][1]], [affine[1][0], affine[1][1]]];
    let mut out = [[0.0; 2]; 2];
    for (r, row) in out.iter_mut().enumerate() {
        for (c, value) in row.iter_mut().enumerate() {
            *value = a[r][0] * shape[0][c] + a[r][1] * shape[1][c];
        }
    }
    out
}
