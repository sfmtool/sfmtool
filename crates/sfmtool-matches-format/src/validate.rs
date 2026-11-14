// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The structural checks of a `.matches` file, written once for the writer,
//! the reader and the verifier.
//!
//! The content checks work on plain slices and push every finding onto a
//! list, so [`crate::verify_matches`] reports them all from the raw bytes it
//! reads, while [`crate::write_matches`] and [`crate::read_matches`] run
//! [`content_errors`] over a [`MatchesData`] and stop at the first. The rules
//! are those of `specs/formats/matches-file-format.md` § "Data Ordering and
//! Constraints".
//!
//! Each content check expects every slice to be the length its declared count
//! calls for, which each caller establishes first: the writer through
//! [`validate_dimensions`], the reader by reading each entry at the length its
//! name states, and the verifier through its byte-length checks. The checks
//! still index only through `get` or within lengths they have checked
//! themselves, so a slice of the wrong length yields wrong findings, never a
//! panic. Within one family of checks (the pair order, the CSR offsets, the
//! references) each check stops at its first finding.
//!
//! [`validate_structure`] and [`validate_dimensions`] check what only an
//! in-memory [`MatchesData`] can get wrong, the sections and flags supplied
//! and the array shapes, and so serve the writer alone; the reader and the
//! verifier check the same things against the archive's entries.

use std::borrow::Cow;

use ndarray::{ArrayBase, Data, Dimension};

use crate::types::*;

/// Return violations of the backbone-specific metadata count rule.
///
/// The required-count error precedes the forbidden-count error so callers
/// that accumulate diagnostics report the same stable order as the writer.
pub(crate) fn check_backbone_counts(metadata: &MatchesMetadata) -> Vec<String> {
    let mut errors = Vec::new();
    if metadata.has_clusters {
        if metadata.cluster_count.is_none() || metadata.cluster_member_count.is_none() {
            errors.push(
                "cluster-bearing file requires metadata.cluster_count and metadata.cluster_member_count"
                    .into(),
            );
        }
        if metadata.image_pair_count.is_some() || metadata.match_count.is_some() {
            errors.push(
                "cluster-bearing file must not set metadata.image_pair_count / match_count".into(),
            );
        }
    } else {
        if metadata.image_pair_count.is_none() || metadata.match_count.is_none() {
            errors.push(
                "pairwise file requires metadata.image_pair_count and metadata.match_count".into(),
            );
        }
        if metadata.cluster_count.is_some() || metadata.cluster_member_count.is_some() {
            errors.push(
                "pairwise file must not set metadata.cluster_count / cluster_member_count".into(),
            );
        }
    }
    errors
}

/// Check that no `image_dims` value is zero. `image_dims` is the `(N, 2)`
/// width/height table, row-major.
pub(crate) fn image_dims_errors(image_dims: &[u32], errors: &mut Vec<String>) {
    if let Some(k) = image_dims.iter().position(|&v| v == 0) {
        errors.push(format!(
            "image_dims[{}] has a zero {} (every dimension must be >= 1)",
            k / 2,
            if k % 2 == 0 { "width" } else { "height" }
        ));
    }
}

/// Check the pairwise backbone's pair table: the pairs sorted with
/// `idx_i < idx_j` and within `image_count`, the per-pair `match_counts` at
/// least 1 and summing to the match count, and every matched feature index
/// within its image's `feature_counts` entry.
///
/// `image_index_pairs` is `(P, 2)` and `match_feature_indexes` `(M, 2)`,
/// row-major. The feature-bounds walk runs only when the counts sum to `M`,
/// since they are the run lengths it advances through.
pub(crate) fn pair_errors(
    image_index_pairs: &[u32],
    match_counts: &[u32],
    match_feature_indexes: &[u32],
    feature_counts: &[u32],
    image_count: usize,
    errors: &mut Vec<String>,
) {
    let pairs = image_index_pairs.as_chunks::<2>().0;
    let matches = match_feature_indexes.as_chunks::<2>().0;

    for (k, &[idx_i, idx_j]) in pairs.iter().enumerate() {
        if idx_i >= idx_j {
            errors.push(format!(
                "image_index_pairs[{k}] = ({idx_i}, {idx_j}): idx_i must be < idx_j"
            ));
            break;
        }
        if idx_i as usize >= image_count || idx_j as usize >= image_count {
            errors.push(format!(
                "image_index_pairs[{k}] = ({idx_i}, {idx_j}): index out of bounds (image_count = {image_count})"
            ));
            break;
        }
        if k > 0 {
            let [prev_i, prev_j] = pairs[k - 1];
            if (idx_i, idx_j) <= (prev_i, prev_j) {
                errors.push(format!(
                    "image_index_pairs not sorted at index {k}: ({prev_i}, {prev_j}) >= ({idx_i}, {idx_j})"
                ));
                break;
            }
        }
    }

    let match_count = matches.len();
    let sum: u64 = match_counts.iter().map(|&c| u64::from(c)).sum();
    let sum_ok = sum == match_count as u64;
    if !sum_ok {
        errors.push(format!(
            "sum of match_counts ({sum}) != match_count ({match_count})"
        ));
    }
    if match_counts.contains(&0) {
        errors.push(
            "match_counts contains values < 1 (pairs with zero matches should not be stored)"
                .into(),
        );
    }
    if !sum_ok {
        return;
    }

    let mut offset = 0usize;
    'pairs: for (&[idx_i, idx_j], &count) in pairs.iter().zip(match_counts) {
        let (Some(&fc_i), Some(&fc_j)) = (
            feature_counts.get(idx_i as usize),
            feature_counts.get(idx_j as usize),
        ) else {
            // An image index out of range, reported by the order check above.
            break;
        };
        let end = offset + count as usize;
        for (m, &[fi, fj]) in matches[offset..end].iter().enumerate() {
            let m = offset + m;
            if fi >= fc_i {
                errors.push(format!(
                    "match_feature_indexes[{m}][0] = {fi} >= feature_counts[{idx_i}] = {fc_i}"
                ));
                break 'pairs;
            }
            if fj >= fc_j {
                errors.push(format!(
                    "match_feature_indexes[{m}][1] = {fj} >= feature_counts[{idx_j}] = {fc_j}"
                ));
                break 'pairs;
            }
        }
        offset = end;
    }
}

/// Check the two-view geometries' inliers against the pair table they
/// verify: the per-pair `inlier_counts` summing to the inlier count, and every
/// inlier one of its pair's candidate matches.
///
/// `inlier_feature_indexes` is `(I, 2)`, row-major, and `candidates` the pair
/// table's `match_counts` and `(M, 2)` `match_feature_indexes`, or `None` when
/// the caller could not read it, which skips the subset walk. The walk also
/// runs only when both count arrays sum to their totals and have one entry per
/// pair; a match-count sum that does not is [`pair_errors`]'s finding.
pub(crate) fn inlier_errors(
    inlier_counts: &[u32],
    inlier_feature_indexes: &[u32],
    candidates: Option<(&[u32], &[u32])>,
    errors: &mut Vec<String>,
) {
    let inliers = inlier_feature_indexes.as_chunks::<2>().0;

    let inlier_count = inliers.len();
    let inlier_sum: u64 = inlier_counts.iter().map(|&c| u64::from(c)).sum();
    if inlier_sum != inlier_count as u64 {
        errors.push(format!(
            "sum of inlier_counts ({inlier_sum}) != inlier_count ({inlier_count})"
        ));
        return;
    }
    let Some((match_counts, match_feature_indexes)) = candidates else {
        return;
    };
    let matches = match_feature_indexes.as_chunks::<2>().0;
    let match_sum: u64 = match_counts.iter().map(|&c| u64::from(c)).sum();
    if match_sum != matches.len() as u64 || inlier_counts.len() != match_counts.len() {
        return;
    }

    // One pair's candidates, sorted for a binary search per inlier, reused
    // across pairs.
    let mut candidates: Vec<[u32; 2]> = Vec::new();
    let mut match_offset = 0usize;
    let mut inlier_offset = 0usize;
    for (k, (&mc, &ic)) in match_counts.iter().zip(inlier_counts).enumerate() {
        let match_end = match_offset + mc as usize;
        let inlier_end = inlier_offset + ic as usize;
        let pair_matches = &matches[match_offset..match_end];
        let pair_inliers = &inliers[inlier_offset..inlier_end];
        match_offset = match_end;
        inlier_offset = inlier_end;
        // Inliers usually keep the order of the candidates they were filtered
        // from, so one merge-walk settles most pairs in linear time; inliers
        // in any other order fall through to the search, which also names the
        // first one that is not a candidate.
        let mut walk = pair_matches.iter();
        if pair_inliers.iter().all(|inlier| walk.any(|m| m == inlier)) {
            continue;
        }
        candidates.clear();
        candidates.extend_from_slice(pair_matches);
        candidates.sort_unstable();
        for (i, pair) in pair_inliers.iter().enumerate() {
            if candidates.binary_search(pair).is_err() {
                let [fi, fj] = *pair;
                errors.push(format!(
                    "inlier ({fi}, {fj}) at index {} for pair {k} is not in candidate matches",
                    inlier_end - pair_inliers.len() + i
                ));
                return;
            }
        }
    }
}

/// Check that every `config_indexes` entry names one of the
/// `config_type_count` entries of `config_types`.
pub(crate) fn config_index_errors(
    config_indexes: &[u8],
    config_type_count: usize,
    errors: &mut Vec<String>,
) {
    if let Some((k, &idx)) = config_indexes
        .iter()
        .enumerate()
        .find(|&(_, &idx)| idx as usize >= config_type_count)
    {
        errors.push(format!(
            "config_indexes[{k}] = {idx} >= config_types length {config_type_count}"
        ));
    }
}

/// The cluster backbone's arrays, as the content checks read them.
pub(crate) struct ClusterSlices<'a> {
    /// `(C + 1,)` CSR offsets into the member arrays.
    pub(crate) cluster_starts: &'a [u32],
    /// `(K,)` each member's image index.
    pub(crate) member_images: &'a [u32],
    /// `(K,)` each member's feature index in its image.
    pub(crate) member_features: &'a [u32],
    /// `(K, 2)` row-major, or `None` to check nothing.
    pub(crate) member_positions: Option<&'a [f32]>,
    /// `(K, 2, 2)` row-major, or `None` to check nothing.
    pub(crate) member_affine_shapes: Option<&'a [f32]>,
}

/// Check the cluster backbone: `cluster_starts` a CSR offset array over the
/// members (leading 0, non-decreasing, every cluster of at least 2 members,
/// ending at the member count), then each member's image within
/// `image_count` and feature within that image's `feature_counts` entry, and
/// no `NaN` in the members' geometry.
///
/// Returns whether the offsets can index the members, which is what the
/// member checks here and [`cluster_patch_errors`] need; they are skipped when
/// it is false.
pub(crate) fn cluster_errors(
    backbone: &ClusterSlices<'_>,
    feature_counts: &[u32],
    image_count: usize,
    errors: &mut Vec<String>,
) -> bool {
    let ClusterSlices {
        cluster_starts,
        member_images,
        member_features,
        member_positions,
        member_affine_shapes,
    } = *backbone;
    let member_count = member_images.len();
    let Some((&first, _)) = cluster_starts.split_first() else {
        errors.push("cluster_starts is empty (it holds cluster_count + 1 offsets)".into());
        return false;
    };
    if first != 0 {
        errors.push(format!("cluster_starts[0] = {first} != 0"));
        return false;
    }
    for (c, run) in cluster_starts.windows(2).enumerate() {
        let (start, end) = (run[0], run[1]);
        if end < start {
            errors.push(format!(
                "cluster_starts not non-decreasing at cluster {c}: {start} > {end}"
            ));
            return false;
        }
        if end - start < 2 {
            errors.push(format!(
                "cluster {c} has {} members; every cluster must have >= 2",
                end - start
            ));
            return false;
        }
    }
    let last = cluster_starts[cluster_starts.len() - 1];
    if last as usize != member_count {
        errors.push(format!(
            "cluster_starts final value {last} != member count {member_count}"
        ));
        return false;
    }

    for (k, (&img, &feat)) in member_images.iter().zip(member_features).enumerate() {
        let fc = match feature_counts.get(img as usize) {
            Some(&fc) if (img as usize) < image_count => fc,
            _ => {
                errors.push(format!(
                    "member_images[{k}] = {img} >= image_count {image_count}"
                ));
                break;
            }
        };
        if feat >= fc {
            errors.push(format!(
                "member_features[{k}] = {feat} >= feature_counts[{img}] = {fc}"
            ));
            break;
        }
    }

    for (name, values, per_member) in [
        ("member_positions", member_positions, 2),
        ("member_affine_shapes", member_affine_shapes, 4),
    ] {
        if let Some(i) = values.and_then(|v| v.iter().position(|v| v.is_nan())) {
            errors.push(format!("{name}[{}] contains NaN", i / per_member));
        }
    }
    true
}

/// Check `cluster_patches/` against the cluster backbone it enriches: every
/// `member_status` a [`ClusterMemberStatus`] discriminant, each refinable
/// cluster's reference member inside the cluster, marked `reference` and with
/// a non-singular affine shape, and at most one `reference`-or-`kept` member
/// per (cluster, image).
///
/// `member_status` is in the canonical numbering, and `backbone` must have
/// passed [`cluster_errors`]. Without `member_affine_shapes` the singularity
/// check is skipped. The reference and coverage checks stop at the first
/// cluster that breaks one.
pub(crate) fn cluster_patch_errors(
    backbone: &ClusterSlices<'_>,
    reference_members: &[u32],
    member_status: &[u8],
    errors: &mut Vec<String>,
) {
    let ClusterSlices {
        cluster_starts,
        member_images,
        member_affine_shapes,
        ..
    } = *backbone;
    if let Some((k, &status)) = member_status
        .iter()
        .enumerate()
        .find(|(_, &s)| ClusterMemberStatus::from_u8(s).is_none())
    {
        errors.push(format!(
            "member_status[{k}] = {status} is not a valid ClusterMemberStatus discriminant"
        ));
    }

    let reference_code = ClusterMemberStatus::Reference as u8;
    let kept_code = ClusterMemberStatus::Kept as u8;
    let mut covering: Vec<(u32, usize)> = Vec::new();
    for (c, (run, &reference)) in cluster_starts.windows(2).zip(reference_members).enumerate() {
        let (start, end) = (run[0], run[1]);
        if reference != CLUSTER_REFERENCE_UNREFINABLE {
            if reference < start || reference >= end {
                errors.push(format!(
                    "reference_members[{c}] = {reference} outside cluster member range [{start}, {end})"
                ));
                break;
            }
            let k = reference as usize;
            let status = member_status.get(k).copied();
            if status != Some(reference_code) {
                errors.push(format!(
                    "reference_members[{c}] = {reference} has status {}, expected {reference_code} (reference)",
                    status.map_or_else(|| "missing".to_string(), |s| s.to_string())
                ));
                break;
            }
            // A cluster's reference member carries `S_ref`, its own detector
            // affine shape, in the backbone's shape array. The value is not
            // checkable without the `.sift` data, but it must be invertible:
            // every consumer that wants a reference-relative warp recovers it
            // as `W = S·S_ref⁻¹` through that member's shape.
            if let Some(s) = member_affine_shapes.and_then(|s| s.get(k * 4..k * 4 + 4)) {
                let det = f64::from(s[0]) * f64::from(s[3]) - f64::from(s[1]) * f64::from(s[2]);
                if !det.is_finite() || det == 0.0 {
                    errors.push(format!(
                        "member_affine_shapes[{k}] (cluster {c}'s reference member) is singular \
                         (det {det})"
                    ));
                    break;
                }
            }
        }

        // At most one Reference/Kept member per (cluster, image). The
        // cluster's covering members are sorted by (image, member), so a
        // second member on an image follows the first; the one reported is
        // the earliest such second member, the first a scan in member order
        // would meet.
        covering.clear();
        for m in start as usize..end as usize {
            let (Some(&status), Some(&img)) = (member_status.get(m), member_images.get(m)) else {
                break;
            };
            if status == reference_code || status == kept_code {
                covering.push((img, m));
            }
        }
        covering.sort_unstable();
        if let Some(pair) = covering
            .windows(2)
            .filter(|pair| pair[0].0 == pair[1].0)
            .min_by_key(|pair| pair[1].1)
        {
            let ((img, prev), (_, m)) = (pair[0], pair[1]);
            errors.push(format!(
                "cluster {c}: members {prev} and {m} are both reference/kept for image {img}"
            ));
            break;
        }
    }
}

/// The contiguous row-major elements of `array`, borrowed when it is already
/// in standard layout.
fn elements<A, S, D>(array: &ArrayBase<S, D>) -> Cow<'_, [A]>
where
    A: Clone,
    S: Data<Elem = A>,
    D: Dimension,
{
    match array.as_slice() {
        Some(slice) => Cow::Borrowed(slice),
        None => Cow::Owned(array.iter().cloned().collect()),
    }
}

/// The findings of every content check above on `data`, in the order the
/// checks run: image dimensions, then the pair table and the two-view
/// geometries over it, or the cluster backbone, its patches and their per-cell
/// columns ([`MemberCellData`](crate::MemberCellData)'s own check).
///
/// For a `data` whose arrays have the shapes its counts declare, as the writer
/// checks with [`validate_dimensions`] and the reader guarantees by reading
/// each entry at its stated length.
pub(crate) fn content_errors(data: &MatchesData) -> Vec<String> {
    let mut errors = Vec::new();
    let image_count = data.metadata.image_count as usize;
    let feature_counts = elements(&data.feature_counts);

    if let Some(image_dims) = &data.image_dims {
        image_dims_errors(&elements(image_dims), &mut errors);
    }

    if let Some(pairs) = &data.image_pairs {
        let match_counts = elements(&pairs.match_counts);
        let match_feature_indexes = elements(&pairs.match_feature_indexes);
        pair_errors(
            &elements(&pairs.image_index_pairs),
            &match_counts,
            &match_feature_indexes,
            &feature_counts,
            image_count,
            &mut errors,
        );
        if let Some(tvg) = &data.two_view_geometries {
            inlier_errors(
                &elements(&tvg.inlier_counts),
                &elements(&tvg.inlier_feature_indexes),
                Some((&match_counts, &match_feature_indexes)),
                &mut errors,
            );
            config_index_errors(
                &elements(&tvg.config_indexes),
                tvg.config_types.len(),
                &mut errors,
            );
        }
    }

    if let Some(clusters) = &data.clusters {
        let cluster_starts = elements(&clusters.cluster_starts);
        let member_images = elements(&clusters.member_images);
        let member_features = elements(&clusters.member_features);
        let member_positions = clusters.member_positions.as_ref().map(elements);
        let member_affine_shapes = clusters.member_affine_shapes.as_ref().map(elements);
        let backbone = ClusterSlices {
            cluster_starts: &cluster_starts,
            member_images: &member_images,
            member_features: &member_features,
            member_positions: member_positions.as_deref(),
            member_affine_shapes: member_affine_shapes.as_deref(),
        };
        let consistent = cluster_errors(&backbone, &feature_counts, image_count, &mut errors);
        if let (true, Some(cp)) = (consistent, &data.cluster_patches) {
            let member_status = elements(&cp.member_status);
            cluster_patch_errors(
                &backbone,
                &elements(&cp.reference_members),
                &member_status,
                &mut errors,
            );
            if let Some(error) = cp
                .member_cells
                .as_ref()
                .and_then(|cells| cells.validation_error(&member_status))
            {
                errors.push(error);
            }
        }
    }

    errors
}

/// Validate the backbone rule, section dependencies, and metadata flag /
/// summary-count consistency of the data handed to the writer.
pub(crate) fn validate_structure(data: &MatchesData) -> Result<(), MatchesError> {
    macro_rules! invalid {
        ($cond:expr, $msg:expr) => {
            if $cond {
                return Err(MatchesError::InvalidFormat($msg.into()));
            }
        };
    }

    let has_pairs = data.image_pairs.is_some();
    let has_clusters = data.clusters.is_some();

    invalid!(
        has_pairs && has_clusters,
        "exactly one of image_pairs / clusters must be present (both supplied)"
    );
    invalid!(
        !has_pairs && !has_clusters,
        "exactly one of image_pairs / clusters must be present (neither supplied)"
    );
    invalid!(
        data.cluster_patches.is_some() && !has_clusters,
        "cluster_patches requires the clusters section"
    );
    invalid!(
        data.two_view_geometries.is_some() && !has_pairs,
        "two_view_geometries requires the image_pairs section"
    );

    // Metadata flags must agree with the supplied sections.
    invalid!(
        data.metadata.has_clusters != has_clusters,
        format!(
            "metadata.has_clusters ({}) doesn't match clusters presence ({has_clusters})",
            data.metadata.has_clusters
        )
    );
    invalid!(
        data.metadata.has_cluster_patches != data.cluster_patches.is_some(),
        format!(
            "metadata.has_cluster_patches ({}) doesn't match cluster_patches presence ({})",
            data.metadata.has_cluster_patches,
            data.cluster_patches.is_some()
        )
    );
    invalid!(
        data.metadata.has_two_view_geometries != data.two_view_geometries.is_some(),
        format!(
            "metadata.has_two_view_geometries ({}) doesn't match two_view_geometries presence ({})",
            data.metadata.has_two_view_geometries,
            data.two_view_geometries.is_some()
        )
    );

    // Backbone-specific summary counts: pairwise files carry
    // image_pair_count / match_count, cluster files carry cluster_count /
    // cluster_member_count — never both.
    if let Some(error) = check_backbone_counts(&data.metadata).into_iter().next() {
        return Err(MatchesError::InvalidFormat(error));
    }

    Ok(())
}

/// Check that every array the writer is handed has the shape the metadata's
/// counts declare, and that the arrays a format version made mandatory are
/// present.
pub(crate) fn validate_dimensions(
    data: &MatchesData,
    image_count: usize,
) -> Result<(), MatchesError> {
    macro_rules! check {
        ($cond:expr, $msg:expr) => {
            if !($cond) {
                return Err(MatchesError::ShapeMismatch($msg.into()));
            }
        };
    }

    // Images
    check!(
        data.image_names.len() == image_count,
        format!(
            "image_names count {} != image_count {image_count}",
            data.image_names.len()
        )
    );
    check!(
        data.feature_tool_hashes.len() == image_count,
        format!(
            "feature_tool_hashes len {} != image_count {image_count}",
            data.feature_tool_hashes.len()
        )
    );
    check!(
        data.sift_content_hashes.len() == image_count,
        format!(
            "sift_content_hashes len {} != image_count {image_count}",
            data.sift_content_hashes.len()
        )
    );
    check!(
        data.feature_counts.len() == image_count,
        format!(
            "feature_counts len {} != image_count {image_count}",
            data.feature_counts.len()
        )
    );
    let Some(image_dims) = &data.image_dims else {
        return Err(MatchesError::ShapeMismatch(
            "image_dims is required (mandatory since format version 4): supply an (N, 2) \
             width/height array"
                .into(),
        ));
    };
    check!(
        image_dims.shape() == [image_count, 2],
        format!(
            "image_dims shape {:?} != [{image_count}, 2]",
            image_dims.shape()
        )
    );

    // Image pairs
    if let Some(pairs) = &data.image_pairs {
        let pair_count = data.metadata.image_pair_count.unwrap_or(0) as usize;
        let match_count = data.metadata.match_count.unwrap_or(0) as usize;

        check!(
            pairs.image_index_pairs.shape() == [pair_count, 2],
            format!(
                "image_index_pairs shape {:?} != [{pair_count}, 2]",
                pairs.image_index_pairs.shape()
            )
        );
        check!(
            pairs.match_counts.len() == pair_count,
            format!(
                "match_counts len {} != pair_count {pair_count}",
                pairs.match_counts.len()
            )
        );
        check!(
            pairs.match_feature_indexes.shape() == [match_count, 2],
            format!(
                "match_feature_indexes shape {:?} != [{match_count}, 2]",
                pairs.match_feature_indexes.shape()
            )
        );
        check!(
            pairs.match_descriptor_distances.len() == match_count,
            format!(
                "match_descriptor_distances len {} != match_count {match_count}",
                pairs.match_descriptor_distances.len()
            )
        );
    }

    // Clusters
    if let Some(clusters) = &data.clusters {
        let cluster_count = data.metadata.cluster_count.unwrap_or(0) as usize;
        let member_count = data.metadata.cluster_member_count.unwrap_or(0) as usize;

        check!(
            clusters.cluster_starts.len() == cluster_count + 1,
            format!(
                "cluster_starts len {} != cluster_count + 1 ({})",
                clusters.cluster_starts.len(),
                cluster_count + 1
            )
        );
        check!(
            clusters.member_images.len() == member_count,
            format!(
                "member_images len {} != cluster_member_count {member_count}",
                clusters.member_images.len()
            )
        );
        check!(
            clusters.member_features.len() == member_count,
            format!(
                "member_features len {} != cluster_member_count {member_count}",
                clusters.member_features.len()
            )
        );
        // The members' detections, mandatory since format version 6. A file
        // that predates them cannot gain them here: they live in the `.sift`
        // rows `member_features` names, which this crate never opens.
        let (Some(member_positions), Some(member_affine_shapes)) =
            (&clusters.member_positions, &clusters.member_affine_shapes)
        else {
            return Err(MatchesError::ShapeMismatch(
                "clusters.member_positions and clusters.member_affine_shapes are required \
                 (mandatory since format version 6): supply the members' detected keypoint \
                 positions (M, 2) and affine shapes (M, 2, 2), copied from the referenced \
                 `.sift` files, or re-run `sfm match --cluster` to regenerate the file"
                    .into(),
            ));
        };
        check!(
            member_positions.shape() == [member_count, 2],
            format!(
                "member_positions shape {:?} != [{member_count}, 2]",
                member_positions.shape()
            )
        );
        check!(
            member_affine_shapes.shape() == [member_count, 2, 2],
            format!(
                "member_affine_shapes shape {:?} != [{member_count}, 2, 2]",
                member_affine_shapes.shape()
            )
        );
    }

    // Cluster patches
    if let Some(cp) = &data.cluster_patches {
        let cluster_count = data.metadata.cluster_count.unwrap_or(0) as usize;
        let member_count = data.metadata.cluster_member_count.unwrap_or(0) as usize;

        check!(
            cp.reference_members.len() == cluster_count,
            format!(
                "reference_members len {} != cluster_count {cluster_count}",
                cp.reference_members.len()
            )
        );
        check!(
            cp.member_status.len() == member_count,
            format!(
                "member_status len {} != cluster_member_count {member_count}",
                cp.member_status.len()
            )
        );
        check!(
            cp.member_zncc.len() == member_count,
            format!(
                "member_zncc len {} != cluster_member_count {member_count}",
                cp.member_zncc.len()
            )
        );
        check!(
            cp.member_consistency_residual.len() == member_count,
            format!(
                "member_consistency_residual len {} != cluster_member_count {member_count}",
                cp.member_consistency_residual.len()
            )
        );
        check!(
            cp.member_shift_px.len() == member_count,
            format!(
                "member_shift_px len {} != cluster_member_count {member_count}",
                cp.member_shift_px.len()
            )
        );
        if let Some(cells) = &cp.member_cells {
            for (name, shape, expected) in [
                (
                    "member_cell_shift_px",
                    cells.shift_px.shape(),
                    &[member_count, 3, 3, 2][..],
                ),
                (
                    "member_cell_zncc",
                    cells.zncc.shape(),
                    &[member_count, 3, 3],
                ),
                (
                    "member_cell_status",
                    cells.status.shape(),
                    &[member_count, 3, 3],
                ),
                (
                    "member_cell_iterations",
                    cells.iterations.shape(),
                    &[member_count],
                ),
            ] {
                check!(
                    shape == expected,
                    format!("{name} shape {shape:?} != {expected:?} (cluster_member_count {member_count})")
                );
            }
        }
    }

    // TVG dimensions
    if let Some(tvg) = &data.two_view_geometries {
        let pair_count = data.metadata.image_pair_count.unwrap_or(0) as usize;
        let inlier_count = tvg.metadata.inlier_count as usize;

        check!(
            tvg.metadata.image_pair_count as usize == pair_count,
            format!(
                "TVG image_pair_count {} != pair_count {pair_count}",
                tvg.metadata.image_pair_count
            )
        );
        check!(
            tvg.config_indexes.len() == pair_count,
            format!(
                "config_indexes len {} != pair_count {pair_count}",
                tvg.config_indexes.len()
            )
        );
        check!(
            tvg.inlier_counts.len() == pair_count,
            format!(
                "inlier_counts len {} != pair_count {pair_count}",
                tvg.inlier_counts.len()
            )
        );
        check!(
            tvg.inlier_feature_indexes.shape() == [inlier_count, 2],
            format!(
                "inlier_feature_indexes shape {:?} != [{inlier_count}, 2]",
                tvg.inlier_feature_indexes.shape()
            )
        );
        check!(
            tvg.f_matrices.shape() == [pair_count, 3, 3],
            format!(
                "f_matrices shape {:?} != [{pair_count}, 3, 3]",
                tvg.f_matrices.shape()
            )
        );
        check!(
            tvg.e_matrices.shape() == [pair_count, 3, 3],
            format!(
                "e_matrices shape {:?} != [{pair_count}, 3, 3]",
                tvg.e_matrices.shape()
            )
        );
        check!(
            tvg.h_matrices.shape() == [pair_count, 3, 3],
            format!(
                "h_matrices shape {:?} != [{pair_count}, 3, 3]",
                tvg.h_matrices.shape()
            )
        );
        check!(
            tvg.quaternions_wxyz.shape() == [pair_count, 4],
            format!(
                "quaternions_wxyz shape {:?} != [{pair_count}, 4]",
                tvg.quaternions_wxyz.shape()
            )
        );
        check!(
            tvg.translations_xyz.shape() == [pair_count, 3],
            format!(
                "translations_xyz shape {:?} != [{pair_count}, 3]",
                tvg.translations_xyz.shape()
            )
        );
    }

    Ok(())
}
