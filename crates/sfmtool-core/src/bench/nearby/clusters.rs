// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The cluster-patches clusters near a pixel, vetted by triangulating their
//! members, as candidate tracks.
//!
//! `specs/core/bench/nearby-sources.md` is the design.

use crate::bench::track_at_pixel::{
    MatchesClusters, ViewCamera, STATUS_KEPT, STATUS_NOT_EVALUATED, STATUS_REFERENCE,
    STATUS_REJECTED_LOW_ZNCC, STATUS_REJECTED_SHIFT, STATUS_REJECTED_UNLOCALIZABLE_CELLS,
    STATUS_REJECTED_UNLOCALIZABLE_REFINED,
};
use crate::patch::normal_refine::ProjectedImage;

use super::candidate::{candidate, check_query, NearbyCandidate, NearbySource, NearbySourceError};
use super::triangulate::meet_dropping_worst;

/// Which of a cluster's members a candidate may be built from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClusterMembers {
    /// The reference and the members the cluster-patches refinement kept.
    Kept,
    /// Every member in a reconstruction image that carries a measurement or
    /// was never evaluated, letting the triangulation drop the bad ones: the
    /// reference, the kept, the rejected for a low ZNCC, a large shift, or
    /// either gate at the refined shape (`rejected_unlocalizable_refined`,
    /// `rejected_unlocalizable_cells`), which were measured before they were
    /// refused, and the unevaluated. A duplicate, and a member refused as
    /// unlocalizable at its detection, are left out.
    Any,
}

impl ClusterMembers {
    /// The policy's name, as the harness spells it (`cluster_members`).
    pub fn name(self) -> &'static str {
        match self {
            Self::Kept => "kept",
            Self::Any => "any",
        }
    }

    /// Whether a member with the `.matches` member status `status` may be used.
    fn admits(self, status: u8) -> bool {
        match self {
            Self::Kept => matches!(status, STATUS_REFERENCE | STATUS_KEPT),
            Self::Any => matches!(
                status,
                STATUS_REFERENCE
                    | STATUS_KEPT
                    | STATUS_REJECTED_LOW_ZNCC
                    | STATUS_REJECTED_SHIFT
                    | STATUS_REJECTED_UNLOCALIZABLE_REFINED
                    | STATUS_REJECTED_UNLOCALIZABLE_CELLS
                    | STATUS_NOT_EVALUATED
            ),
        }
    }
}

impl std::str::FromStr for ClusterMembers {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "kept" => Ok(Self::Kept),
            "any" => Ok(Self::Any),
            other => Err(format!(
                "unknown cluster members {other:?} (expected kept|any)"
            )),
        }
    }
}

/// What [`nearby_cluster_tracks`] keeps. The defaults are the harness's.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ClusterTracksOptions {
    /// How far from the pixel a cluster's member in the queried image may be,
    /// in px (harness `cluster_radius_px`).
    pub radius_px: f64,
    /// The most clusters tried, the nearest (harness `cluster_max`).
    pub max_clusters: usize,
    /// The largest reprojection error any member may have, in px (harness
    /// `max_reproj_px`).
    pub max_reproj_px: f64,
    /// Which members may be used (harness `cluster_members`).
    pub members: ClusterMembers,
}

impl Default for ClusterTracksOptions {
    fn default() -> Self {
        Self {
            radius_px: 48.0,
            max_clusters: 16,
            max_reproj_px: 2.0,
            members: ClusterMembers::Any,
        }
    }
}

/// The clusters with a member within [`ClusterTracksOptions::radius_px`] of
/// `pixel` in `image`, nearest first, each vetted by triangulating its members,
/// as candidate tracks.
///
/// Up to [`ClusterTracksOptions::max_clusters`] clusters are tried. A cluster's
/// member in `image` is its nearest there, and must be one
/// [`ClusterTracksOptions::members`] admits. In every other image the cluster
/// contributes one admitted member: the reference or a kept one first, then
/// the one whose ZNCC against the reference is highest. The members are
/// triangulated, and the worst is dropped while three or more remain, until
/// every one is within [`ClusterTracksOptions::max_reproj_px`]; a cluster whose
/// worst is the queried member, or that is left with fewer than two, gives no
/// candidate. Each candidate names its cluster in [`NearbyCandidate::id`], and
/// sits at its member in `image`.
///
/// `views` holds one entry per image of the reconstruction, and `clusters` is
/// indexed onto the same images; only the views' cameras are read.
pub fn nearby_cluster_tracks(
    views: &[ProjectedImage<'_>],
    clusters: &MatchesClusters,
    image: u32,
    pixel: [f64; 2],
    options: &ClusterTracksOptions,
) -> Result<Vec<NearbyCandidate>, NearbySourceError> {
    check_query(
        views.len(),
        &[("clusters", clusters.image_count())],
        views,
        image,
        pixel,
    )?;
    let cameras: Vec<ViewCamera<'_>> = views.iter().map(ViewCamera::new).collect();
    let policy = options.members;
    let mut out = Vec::new();
    for near in clusters
        .near(image, pixel, options.radius_px)
        .into_iter()
        .take(options.max_clusters)
    {
        if !policy.admits(clusters.member(near.member).status) {
            continue;
        }
        // One member per image, in the order the images are first met: the
        // queried image's own, and elsewhere the best-reading admitted one.
        let mut best: Vec<(u32, usize, f64)> = Vec::new();
        for k in near.members.clone() {
            let m = clusters.member(k);
            let Some(m_image) = m.image else { continue };
            if !policy.admits(m.status) || (m_image == image && k != near.member) {
                continue;
            }
            let mut z = if m.zncc.is_finite() { m.zncc } else { 0.0 };
            if m.is_kept_or_reference() {
                z += 2.0;
            }
            match best.iter_mut().find(|b| b.0 == m_image) {
                Some(b) => {
                    if z > b.2 {
                        b.1 = k;
                        b.2 = z;
                    }
                }
                None => best.push((m_image, k, z)),
            }
        }
        let sightings: Vec<(u32, [f64; 2])> = best
            .iter()
            .map(|&(i, k, _)| (i, clusters.member(k).position))
            .collect();
        let Some((mut sightings, met)) =
            meet_dropping_worst(&cameras, image, sightings, options.max_reproj_px)
        else {
            continue;
        };
        // The queried member goes first.
        let mut errors = met.errors_px;
        let q = sightings
            .iter()
            .position(|s| s.0 == image)
            .expect("the queried member is never dropped");
        let s = sightings.remove(q);
        sightings.insert(0, s);
        let e = errors.remove(q);
        errors.insert(0, e);
        out.push(candidate(
            &cameras,
            NearbySource::Clusters,
            Some(near.cluster),
            met.position,
            sightings,
            errors,
            pixel,
        ));
    }
    Ok(out)
}
