// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The SIFT index's constellation query from a pixel, its carried seed
//! positions triangulated, as candidate tracks.
//!
//! `specs/core/bench/nearby-sources.md` is the design.

use crate::bench::search::{search_descriptors, Found, SearchOptions};
use crate::bench::steps::ClusterSeed;
use crate::bench::track::Thresholds;
use crate::bench::track_at_pixel::{seed_cluster_with, SiftIndexSource, ViewCamera};
use crate::features::kdforest::{radius_for_feature_count, ConstellationParams};
use crate::patch::normal_refine::ProjectedImage;
use crate::progress::Progress;
use crate::reconstruction::edited::EditedReconstruction;

use super::candidate::{candidate, check_query, NearbyCandidate, NearbySource, NearbySourceError};
use super::guided::keypoints_near;
use super::triangulate::meet_dropping_worst;

/// Where the constellation queries are made from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConstellationAt {
    /// From the pixel alone.
    Pixel,
    /// From the pixel, and from each of the keypoints near it
    /// ([`ConstellationSeedOptions::lateral_max`] within
    /// [`ConstellationSeedOptions::lateral_radius_px`]).
    Keypoints,
}

impl ConstellationAt {
    /// The name the harness spells it with (`constellation_at`).
    pub fn name(self) -> &'static str {
        match self {
            Self::Pixel => "pixel",
            Self::Keypoints => "keypoints",
        }
    }
}

impl std::str::FromStr for ConstellationAt {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "pixel" => Ok(Self::Pixel),
            "keypoints" => Ok(Self::Keypoints),
            other => Err(format!(
                "unknown constellation_at {other:?} (expected pixel|keypoints)"
            )),
        }
    }
}

/// What [`constellation_seeds`] asks and keeps. The defaults are the
/// harness's.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConstellationSeedOptions {
    /// The constellation's radius holds about this many of the queried image's
    /// keypoints (harness `constellation_target`).
    pub target: usize,
    /// The fewest agreeing correspondences an image needs (harness
    /// `constellation_min_inliers`).
    pub min_inliers: usize,
    /// The seeded cluster's radius, in px (harness
    /// `constellation_radius_px`); the query reads its position, not its
    /// size.
    pub seed_radius_px: f64,
    /// The largest reprojection error any sighting may have, in px (harness
    /// `constellation_max_reproj_px`): the sightings are the warp's
    /// predictions, not refined.
    pub max_reproj_px: f64,
    /// Where the queries are made from (harness `constellation_at`).
    pub at: ConstellationAt,
    /// The most keypoints queried from, with [`ConstellationAt::Keypoints`]
    /// (harness `lateral_max`).
    pub lateral_max: usize,
    /// How far from the pixel those keypoints may be, in px (harness
    /// `lateral_radius_px`).
    pub lateral_radius_px: f64,
}

impl Default for ConstellationSeedOptions {
    fn default() -> Self {
        Self {
            target: 50,
            min_inliers: 6,
            seed_radius_px: 6.0,
            max_reproj_px: 3.0,
            at: ConstellationAt::Pixel,
            lateral_max: 4,
            lateral_radius_px: 24.0,
        }
    }
}

/// Keypoints this near the pixel or nearer are not queried from with
/// [`ConstellationAt::Keypoints`]: the pixel's own query already covers them.
const LATERAL_SKIP_PX: f64 = 1.0;

/// The SIFT index's constellation query from `pixel` in `image`, and with
/// [`ConstellationAt::Keypoints`] from the keypoints near it, each as a
/// candidate track.
///
/// A query takes the queried image's keypoints within the radius that holds
/// about [`ConstellationSeedOptions::target`] of them, and each other image
/// whose matches agree on one affine warp, with
/// [`ConstellationSeedOptions::min_inliers`] or more, carries the query's
/// position into its own frame by that warp ([`search_descriptors`], seeded
/// with a one-sighting cluster there). Those seed positions and the query's
/// own are triangulated, dropping the worst while three or more remain, until
/// every one is within [`ConstellationSeedOptions::max_reproj_px`]. The
/// cluster refinement is not used to read the seeds, since on a grazing
/// surface it rejects the true matches. A query that finds no other image, or
/// whose sightings do not meet, gives no candidate. The pixel's candidate has
/// no [`NearbyCandidate::id`], and a keypoint's is named by its row.
///
/// `views` holds one entry per image of `edited`, and `index` the same
/// images; `edited` is read for the image's name only.
pub fn constellation_seeds(
    edited: &EditedReconstruction,
    views: &[ProjectedImage<'_>],
    index: &SiftIndexSource<'_>,
    image: u32,
    pixel: [f64; 2],
    options: &ConstellationSeedOptions,
) -> Result<Vec<NearbyCandidate>, NearbySourceError> {
    check_query(
        edited.image_count(),
        &[("views", views.len()), ("keypoints", index.keypoints.len())],
        views,
        image,
        pixel,
    )?;
    let cameras: Vec<ViewCamera<'_>> = views.iter().map(ViewCamera::new).collect();
    let keypoints = &index.keypoints[image as usize];
    let mut from: Vec<(Option<u32>, [f64; 2])> = vec![(None, pixel)];
    if options.at == ConstellationAt::Keypoints {
        for k in keypoints_near(
            keypoints,
            pixel,
            options.lateral_radius_px,
            LATERAL_SKIP_PX,
            options.lateral_max,
        ) {
            let p = keypoints.positions[k];
            from.push((Some(k as u32), [f64::from(p[0]), f64::from(p[1])]));
        }
    }
    let name = &edited.base.image_table.images[image as usize].name;
    let stem = std::path::Path::new(name)
        .file_stem()
        .map_or_else(|| name.clone(), |s| s.to_string_lossy().into_owned());
    let camera = &cameras[image as usize];
    let search = SearchOptions {
        constellation: ConstellationParams {
            min_inliers: options.min_inliers,
            ..ConstellationParams::DEFAULT
        },
        radius_px: radius_for_feature_count(
            camera.width,
            camera.height,
            keypoints.len(),
            options.target,
        ),
        min_inliers: options.min_inliers,
    };
    let mut out = Vec::new();
    for (id, at) in from {
        let seed = ClusterSeed::from_pixel(image, stem.clone(), at, options.seed_radius_px);
        let Ok(track) = seed_cluster_with(&seed, &Thresholds::default()) else {
            continue;
        };
        let Ok((_, report)) = search_descriptors(
            &track,
            0,
            keypoints,
            index.forest,
            &search,
            &Progress::none(),
        ) else {
            continue;
        };
        let mut sightings = vec![(image, at)];
        sightings.extend(
            report
                .matches
                .iter()
                .filter(|m| matches!(m.found, Found::Added { .. }))
                .map(|m| (m.image, m.pixel)),
        );
        if sightings.len() < 2 {
            continue;
        }
        let Some((kept, met)) =
            meet_dropping_worst(&cameras, image, sightings, options.max_reproj_px)
        else {
            continue;
        };
        out.push(candidate(
            &cameras,
            NearbySource::Constellation,
            id,
            met.position,
            kept,
            met.errors_px,
            pixel,
        ));
    }
    Ok(out)
}
