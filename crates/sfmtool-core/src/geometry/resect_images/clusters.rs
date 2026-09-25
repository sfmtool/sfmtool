// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The cluster half of [`super::ResectSource::TracksAndClusters`].
//!
//! Each cluster of a cluster-patches `.matches` file is used as a track of its
//! own. It is never joined to the reconstruction's tracks: it has no point
//! index, it is placed by triangulating its own members, and it reaches the
//! pose estimate beside the tracks as one more 2D–3D pair.
//!
//! A cluster gives a target image one pair when all of these hold:
//!
//! - it has exactly one kept member (`Reference` or `Kept`) in the target;
//! - it has kept members in at least two *non-target* posed images;
//! - those non-target members triangulate, at their images' stored poses,
//!   under the rules the held-out re-triangulation of the tracks uses (at
//!   least two usable rays, in front of every camera, an observable depth);
//! - the triangulated position agrees with every one of those members: it
//!   reprojects, at the member's image's stored pose, within the caller's
//!   pixel threshold of the member's refined position. A position behind a
//!   member's camera, or one that projects outside the member's frame, does
//!   not agree.
//!
//! The pair is the target member's refined position against that triangulated
//! position. Members in any target image never contribute a ray, so the
//! cluster's position is held out from the whole target set exactly as a
//! track's is.
//!
//! A cluster with two or more kept members in the target does not say which of
//! its pixels the point is at, so it contributes nothing to that target.

use std::collections::HashMap;

use nalgebra::Vector3;
use sfmtool_matches_format::{ClusterMemberStatus, MatchesData};

use crate::reconstruction::SfmrReconstruction;

use super::{triangulate_groups, Correspondence, Pose, ResectImageError};

/// Sightings of one cluster as `(image, refined pixel)`.
type Sightings = Vec<(usize, [f64; 2])>;

/// What the clusters gave one target, and what became of the clusters it has
/// a member in.
#[derive(Clone, Debug, Default)]
pub(super) struct TargetClusters {
    /// The target's cluster pairs, as `(cluster id, target pixel, position)`.
    /// The id is the cluster's slot in [`ClusterSupport`] offset past the
    /// reconstruction's point indexes, so it cannot collide with a track.
    pub(super) pairs: Vec<Correspondence>,
    /// Clusters with at least one kept member in this target.
    pub(super) considered: usize,
    /// Of those, the ones the member rules set aside: more than one kept
    /// member in this target, or kept members in fewer than two non-target
    /// posed images.
    pub(super) skipped: usize,
    /// Of those, the ones whose non-target members did not triangulate.
    pub(super) failed: usize,
    /// Of those, the ones whose triangulated position lies farther than the
    /// threshold from one of its own non-target members.
    pub(super) inconsistent: usize,
}

/// The clusters that reached at least one target's estimate.
pub(super) struct ClusterSupport {
    /// The first id a cluster pair carries: the reconstruction's point count.
    pub(super) id_base: usize,
    /// Each placed cluster's non-target kept members as `(image, refined
    /// pixel)`, by slot. The estimate reads them for its covisibility ranking,
    /// as it reads the non-target observations of a track.
    pub(super) members: Vec<Sightings>,
    /// What each target got, by target image index.
    pub(super) targets: HashMap<usize, TargetClusters>,
}

/// Whether a member's refined position is a claim about the world: the
/// cluster's reference, and the members refinement kept.
fn is_kept(status: u8) -> bool {
    matches!(
        ClusterMemberStatus::from_u8(status),
        Some(ClusterMemberStatus::Reference) | Some(ClusterMemberStatus::Kept)
    )
}

/// Gather every target's cluster pairs from `matches`.
///
/// `is_target` and `posed_others` are per reconstruction image. Each cluster is
/// triangulated at most once, however many targets it reaches.
/// `max_residual_px` is the self-consistency threshold: a triangulated cluster
/// with a non-target member farther than this from its reprojection gives no
/// pair.
pub(super) fn cluster_support(
    recon: &SfmrReconstruction,
    targets: &[usize],
    is_target: &[bool],
    posed_others: &[bool],
    matches: &MatchesData,
    max_residual_px: f64,
) -> Result<ClusterSupport, ResectImageError> {
    let clusters = matches.clusters.as_ref().ok_or_else(|| {
        ResectImageError::Clusters("the .matches file has no clusters section".to_string())
    })?;
    let patches = matches.cluster_patches.as_ref().ok_or_else(|| {
        ResectImageError::Clusters(
            "the .matches file has no cluster patches section, so its clusters were never \
             refined"
                .to_string(),
        )
    })?;
    let positions = clusters.member_positions.as_ref().ok_or_else(|| {
        ResectImageError::Clusters("the .matches file carries no member positions".to_string())
    })?;

    // The file's images, by their index in the reconstruction. Names are the
    // only identity the two files share, and one of them may have been written
    // with Windows separators.
    let by_name: HashMap<String, usize> = recon
        .image_table
        .images
        .iter()
        .enumerate()
        .map(|(i, image)| (normalize(&image.name), i))
        .collect();
    let to_recon: Vec<Option<usize>> = matches
        .image_names
        .iter()
        .map(|name| by_name.get(&normalize(name)).copied())
        .collect();

    let id_base = recon.point_set.points.len();
    let mut targets_out: HashMap<usize, TargetClusters> = targets
        .iter()
        .map(|&t| (t, TargetClusters::default()))
        .collect();
    // Per cluster that passed the member rules for some target: its
    // non-target members, and which targets it answers with which pixel.
    let mut candidates: Vec<(Sightings, Sightings)> = Vec::new();

    let starts = &clusters.cluster_starts;
    for c in 0..starts.len().saturating_sub(1) {
        let mut in_targets: Sightings = Vec::new();
        let mut others: Sightings = Vec::new();
        for m in starts[c] as usize..starts[c + 1] as usize {
            if !is_kept(patches.member_status[m]) {
                continue;
            }
            let Some(image) = to_recon
                .get(clusters.member_images[m] as usize)
                .copied()
                .flatten()
            else {
                continue;
            };
            let uv = [f64::from(positions[[m, 0]]), f64::from(positions[[m, 1]])];
            if is_target[image] {
                in_targets.push((image, uv));
            } else if posed_others[image] {
                others.push((image, uv));
            }
        }
        if in_targets.is_empty() {
            continue;
        }
        let mut other_images: Vec<usize> = others.iter().map(|&(image, _)| image).collect();
        other_images.sort_unstable();
        other_images.dedup();

        let mut answers: Sightings = Vec::new();
        let mut seen: Vec<usize> = in_targets.iter().map(|&(image, _)| image).collect();
        seen.sort_unstable();
        seen.dedup();
        for t in seen {
            let counts = targets_out.get_mut(&t).expect("a target");
            counts.considered += 1;
            let mut mine = in_targets.iter().filter(|&&(image, _)| image == t);
            let first = mine.next().expect("the target has a member");
            if mine.next().is_some() || other_images.len() < 2 {
                counts.skipped += 1;
                continue;
            }
            answers.push((t, first.1));
        }
        if !answers.is_empty() {
            candidates.push((others, answers));
        }
    }

    // One triangulation per candidate, at the non-target images' stored poses.
    let groups: Vec<Sightings> = candidates.iter().map(|c| c.0.clone()).collect();
    let no_replacement: Vec<Option<Pose>> = vec![None; recon.image_table.images.len()];
    let placed = triangulate_groups(recon, &groups, &no_replacement);

    let mut members: Vec<Sightings> = Vec::new();
    for ((others, answers), position) in candidates.into_iter().zip(placed) {
        let Some(world) = position else {
            for (t, _) in answers {
                targets_out.get_mut(&t).expect("a target").failed += 1;
            }
            continue;
        };
        if worst_residual(recon, &others, world) > max_residual_px {
            for (t, _) in answers {
                targets_out.get_mut(&t).expect("a target").inconsistent += 1;
            }
            continue;
        }
        let id = id_base + members.len();
        members.push(others);
        for (t, uv) in answers {
            targets_out
                .get_mut(&t)
                .expect("a target")
                .pairs
                .push((id, uv, world));
        }
    }
    Ok(ClusterSupport {
        id_base,
        members,
        targets: targets_out,
    })
}

/// The largest distance, in pixels, between a member of `sightings` and where
/// `world` projects in that member's image ([`member_residual`]).
fn worst_residual(recon: &SfmrReconstruction, sightings: &Sightings, world: [f64; 3]) -> f64 {
    sightings
        .iter()
        .map(|&(image, uv)| member_residual(recon, image, uv, world))
        .fold(0.0, f64::max)
}

/// How far, in pixels, `world` projects from `uv` in image `image` at its
/// stored pose.
///
/// Infinite when the point is behind the camera, when its projection falls
/// outside the frame, or when `uv` is not a finite pixel. "Behind" is judged
/// along the ray through `uv`, as the triangulation judges "in front", so a
/// fisheye wider than 180° keeps the points it can see beside its image plane.
pub(super) fn member_residual(
    recon: &SfmrReconstruction,
    image: usize,
    uv: [f64; 2],
    world: [f64; 3],
) -> f64 {
    let stored = &recon.image_table.images[image];
    let camera = &recon.image_table.cameras[stored.camera_index as usize];
    let local = stored.quaternion_wxyz * Vector3::new(world[0], world[1], world[2])
        + stored.translation_xyz;
    let ray = camera.pixel_to_ray(uv[0], uv[1]);
    let along = local.dot(&Vector3::new(ray[0], ray[1], ray[2]));
    if along.is_nan() || along <= 0.0 {
        return f64::INFINITY;
    }
    match camera.ray_to_pixel([local.x, local.y, local.z]) {
        Some((u, v))
            if (0.0..=f64::from(camera.width)).contains(&u)
                && (0.0..=f64::from(camera.height)).contains(&v) =>
        {
            (u - uv[0]).hypot(v - uv[1])
        }
        _ => f64::INFINITY,
    }
}

/// One spelling of an image path: forward slashes, so a `.matches` file written
/// on Windows joins to a reconstruction written anywhere.
fn normalize(name: &str) -> String {
    name.replace('\\', "/")
}
