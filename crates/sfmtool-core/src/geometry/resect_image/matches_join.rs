// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The matches-backed correspondence source of [`super::resect_image`].
//!
//! A `.matches` file says which features of which images look like the same
//! piece of the world. A reconstruction says which of *its* features are which
//! 3D points. Composing the two gives the target image 2D–3D pairs the
//! reconstruction never assigned to it: match rows carry the target's keypoint
//! to a matched keypoint on some other posed image, and that keypoint's feature
//! index is one the reconstruction may already have turned into a point.
//!
//! This is the same construction the offline non-member resection uses, and it
//! joins through feature indexes — so it needs a `sift_files` reconstruction on
//! one side and either backbone (clusters or pairwise matches) on the other.
//!
//! Where the target's pixel comes from: the refined member position in
//! `cluster_patches/` when the file carries one (that is what the cluster
//! actually claims the feature is at), and the target's own `.sift` detection
//! otherwise.

use std::collections::HashMap;

use matches_format::{ClusterMemberStatus, MatchesData};

use crate::reconstruction::SfmrReconstruction;

use super::{read_sift_positions, Correspondence, ResectImageError};

/// Cluster member statuses whose position is a claim about the world: the
/// cluster's reference, and the members refinement kept. Everything else was
/// rejected or never evaluated, and its row is not evidence.
fn is_kept(status: u8) -> bool {
    matches!(
        ClusterMemberStatus::from_u8(status),
        Some(ClusterMemberStatus::Reference) | Some(ClusterMemberStatus::Kept)
    )
}

/// The target's 2D–3D pairs through the match graph, as
/// `(point index, pixel, world position)`.
///
/// The world position is the point's *stored* position; the caller replaces it
/// with the held-out one for any point the target also observes.
pub(super) fn correspondences(
    recon: &SfmrReconstruction,
    image_index: usize,
    matches: &MatchesData,
    posed: &[bool],
) -> Result<Vec<Correspondence>, ResectImageError> {
    if recon.feature_indexes().is_none() {
        return Err(ResectImageError::Matches(
            "matches join needs a sift_files reconstruction: an embedded-patches \
             reconstruction carries no feature indexes to match rows against"
                .to_string(),
        ));
    }

    // Names are the only identity the two files share, and one of them may have
    // been written with Windows separators.
    let by_name: HashMap<String, usize> = matches
        .image_names
        .iter()
        .enumerate()
        .map(|(i, name)| (normalize(name), i))
        .collect();
    let target = *by_name
        .get(&normalize(&recon.images[image_index].name))
        .ok_or_else(|| {
            ResectImageError::Matches(format!(
                "{} is not one of the {} images of this .matches file",
                recon.images[image_index].name,
                matches.image_names.len()
            ))
        })?;
    // Every other posed image of the reconstruction, by its index in the
    // matches file. Images the matches file does not name simply contribute
    // nothing.
    let mut to_recon: HashMap<usize, usize> = HashMap::new();
    for (i, image) in recon.images.iter().enumerate() {
        if i == image_index || !posed[i] {
            continue;
        }
        if let Some(&g) = by_name.get(&normalize(&image.name)) {
            to_recon.insert(g, i);
        }
    }
    if to_recon.is_empty() {
        return Err(ResectImageError::Matches(
            "no other posed image of this reconstruction appears in the .matches file".to_string(),
        ));
    }

    // `(target feature, matched image, matched feature)` — the raw rows, and
    // the target member positions the file supplied for them.
    let mut rows: Vec<(u32, usize, u32)> = Vec::new();
    let mut positions: HashMap<u32, [f64; 2]> = HashMap::new();
    if let Some(clusters) = &matches.clusters {
        let starts = &clusters.cluster_starts;
        let member_images = &clusters.member_images;
        let member_features = &clusters.member_features;
        let patches = matches.cluster_patches.as_ref();
        for c in 0..starts.len().saturating_sub(1) {
            let range = starts[c] as usize..starts[c + 1] as usize;
            let mut mine: Vec<usize> = Vec::new();
            let mut theirs: Vec<usize> = Vec::new();
            for m in range {
                if let Some(p) = patches {
                    if !is_kept(p.member_status[m]) {
                        continue;
                    }
                }
                let image = member_images[m] as usize;
                if image == target {
                    mine.push(m);
                } else if to_recon.contains_key(&image) {
                    theirs.push(m);
                }
            }
            if mine.is_empty() || theirs.is_empty() {
                continue;
            }
            for &m in &mine {
                let feature = member_features[m];
                if let Some(p) = patches {
                    positions.insert(
                        feature,
                        [p.member_affines[[m, 0, 2]], p.member_affines[[m, 1, 2]]],
                    );
                }
                for &o in &theirs {
                    rows.push((
                        feature,
                        to_recon[&(member_images[o] as usize)],
                        member_features[o],
                    ));
                }
            }
        }
    } else if let Some(pairs) = &matches.image_pairs {
        let mut start = 0usize;
        for p in 0..pairs.image_index_pairs.nrows() {
            let count = pairs.match_counts[p] as usize;
            let (a, b) = (
                pairs.image_index_pairs[[p, 0]] as usize,
                pairs.image_index_pairs[[p, 1]] as usize,
            );
            // Whichever side is the target, the other side is the one that has
            // to resolve to a posed image of the reconstruction.
            let side = if a == target && to_recon.contains_key(&b) {
                Some((0usize, 1usize, b))
            } else if b == target && to_recon.contains_key(&a) {
                Some((1usize, 0usize, a))
            } else {
                None
            };
            if let Some((mine, theirs, other)) = side {
                for m in start..start + count {
                    rows.push((
                        pairs.match_feature_indexes[[m, mine]],
                        to_recon[&other],
                        pairs.match_feature_indexes[[m, theirs]],
                    ));
                }
            }
            start += count;
        }
    } else {
        return Err(ResectImageError::Matches(
            "this .matches file carries no correspondence backbone".to_string(),
        ));
    }
    if rows.is_empty() {
        return Err(ResectImageError::Matches(format!(
            "the .matches file connects {} to no posed image of this reconstruction",
            recon.images[image_index].name
        )));
    }

    // The target's own detections, read only when the file supplied no refined
    // member position for a feature the join actually uses.
    let need_sift = rows.iter().any(|r| !positions.contains_key(&r.0));
    let detections = if need_sift {
        read_sift_positions(recon, image_index)?
    } else {
        Vec::new()
    };

    // One pair per distinct (feature, point) claim. A feature claimed by two
    // *different* points is kept: that conflict is the estimate's to resolve,
    // and suppressing it would hide exactly what a wrong candidate does.
    let mut seen: std::collections::HashSet<(u32, usize)> = std::collections::HashSet::new();
    let mut out = Vec::new();
    for (feature, image, other_feature) in rows {
        let Some(&point) = recon.image_feature_to_point[image].get(&other_feature) else {
            continue;
        };
        let point = point as usize;
        if recon.points[point].is_at_infinity() {
            continue;
        }
        if !seen.insert((feature, point)) {
            continue;
        }
        let uv = match positions.get(&feature) {
            Some(&uv) => uv,
            None => match detections.get(feature as usize) {
                Some(p) => [p[0] as f64, p[1] as f64],
                None => continue,
            },
        };
        let position = recon.points[point].position;
        out.push((point, uv, [position.x, position.y, position.z]));
    }
    // Sorted by point so the estimate's input order is a function of the data
    // rather than of the hash map's iteration order.
    out.sort_by(|a, b| a.0.cmp(&b.0).then(a.1[0].total_cmp(&b.1[0])));
    Ok(out)
}

/// One spelling of an image path: forward slashes, so a `.matches` file written
/// on Windows joins to a reconstruction written anywhere.
fn normalize(name: &str) -> String {
    name.replace('\\', "/")
}
