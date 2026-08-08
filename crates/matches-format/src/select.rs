// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Cluster selection: derive a self-shaped `.matches` subset from a
//! cluster-backbone file.
//!
//! [`MatchesData::select_clusters`] applies a per-member / per-cluster
//! predicate ([`ClusterSelect`]) and produces a new, writable [`MatchesData`]
//! holding only the surviving clusters and members, with images, member
//! indexes, and cluster numbering densely renumbered. The derivation is
//! recorded in the output metadata (`matching_options["cluster_selection"]`).
//! See `specs/formats/matches-file-format.md` § "Cluster Selection (Derived
//! Files)".

use ndarray::{Array1, Array2, Array3};

use crate::types::*;

/// Selection predicate for [`MatchesData::select_clusters`].
///
/// A predicate, not a strategy: it decides which members and clusters
/// survive, and never reorders anything — clusters keep their file order
/// (densely renumbered) and members keep their within-cluster order.
#[derive(Debug, Clone)]
pub struct ClusterSelect {
    /// Minimum number of distinct selected images among a cluster's kept
    /// members for the cluster to survive. Must be ≥ 2 (a written cluster
    /// needs ≥ 2 members).
    pub min_span: u32,
    /// Optional image restriction, by image name (entries of
    /// `images/names.json.zst`). When set, members on any other image are
    /// dropped before the span test, and the output image table becomes
    /// exactly this set (in file order) — including requested images that
    /// end up with zero members. Every requested name must exist in the
    /// source file.
    pub restrict_images: Option<Vec<String>>,
    /// Member statuses that survive selection when the source carries a
    /// `cluster_patches/` section. Ignored (every member is a candidate)
    /// when the source has no `cluster_patches/`.
    pub accepted_statuses: Vec<ClusterMemberStatus>,
}

impl Default for ClusterSelect {
    fn default() -> Self {
        Self {
            min_span: 2,
            restrict_images: None,
            accepted_statuses: vec![ClusterMemberStatus::Reference, ClusterMemberStatus::Kept],
        }
    }
}

impl ClusterSelect {
    /// The selection options as the JSON provenance object recorded in the
    /// derived file's `matching_options["cluster_selection"]` (together with
    /// the source file's content hash, added by `select_clusters`).
    fn provenance(&self, source_content_xxh128: &str) -> serde_json::Value {
        serde_json::json!({
            "source_content_xxh128": source_content_xxh128,
            "min_span": self.min_span,
            "restrict_images": self.restrict_images,
            "accepted_statuses": self
                .accepted_statuses
                .iter()
                .map(|s| s.as_str())
                .collect::<Vec<_>>(),
        })
    }
}

impl MatchesData {
    /// Derive a new cluster-backbone [`MatchesData`] holding only the
    /// clusters and members that pass `opts` — a self-shaped, writable
    /// subset.
    ///
    /// Semantics, in order:
    ///
    /// 1. Clusters whose `reference_members` entry is
    ///    [`CLUSTER_REFERENCE_UNREFINABLE`] in the **source** are dropped
    ///    (only when the source carries `cluster_patches/`).
    /// 2. Per cluster, a member is kept iff its status is in
    ///    `accepted_statuses` (when `cluster_patches/` is present) **and**,
    ///    when `restrict_images` is set, its image is in the restriction.
    /// 3. The cluster survives iff its kept members span at least
    ///    `min_span` distinct (selected) images.
    /// 4. Surviving clusters and members are densely renumbered in source
    ///    order; `reference_members` global indexes are remapped. When a
    ///    surviving cluster's reference member was itself dropped (its image
    ///    outside the restriction), the derived entry is
    ///    [`CLUSTER_REFERENCE_UNREFINABLE`] — in a derived file that
    ///    sentinel means "reference not present in this selection"; the
    ///    kept members still carry absolute positions and warps expressed
    ///    relative to the (absent) reference patch.
    /// 5. When restricted, the image table shrinks to exactly the requested
    ///    images (file order preserved, images with zero members included)
    ///    and all parallel image arrays plus `member_images` are renumbered.
    ///
    /// The output metadata carries the derivation provenance in
    /// `matching_options["cluster_selection"]` (source `content_xxh128` +
    /// the selection options); all other metadata — including the timestamp
    /// — is inherited from the source. The output's `content_hash` is
    /// cleared (recomputed by [`crate::write_matches`]).
    ///
    /// Errors when the source stores the pairwise backbone, when
    /// `min_span < 2`, or when a restriction name is not in the source
    /// image table.
    pub fn select_clusters(&self, opts: &ClusterSelect) -> Result<MatchesData, MatchesError> {
        let clusters = self.clusters.as_ref().ok_or_else(|| {
            MatchesError::InvalidFormat(
                "select_clusters requires the cluster backbone; this file stores image_pairs"
                    .into(),
            )
        })?;
        if opts.min_span < 2 {
            return Err(MatchesError::InvalidFormat(format!(
                "min_span must be >= 2 (every written cluster needs >= 2 members), got {}",
                opts.min_span
            )));
        }

        let n_img = self.image_names.len();

        // Image restriction: old index -> new index (dense, file order).
        // Every requested name must exist; requests are a set (duplicates
        // collapse).
        let image_map: Option<Vec<Option<u32>>> = match &opts.restrict_images {
            None => None,
            Some(names) => {
                let mut requested: Vec<bool> = vec![false; n_img];
                let index_of: std::collections::HashMap<&str, usize> = self
                    .image_names
                    .iter()
                    .enumerate()
                    .map(|(i, n)| (n.as_str(), i))
                    .collect();
                for name in names {
                    let Some(&i) = index_of.get(name.as_str()) else {
                        return Err(MatchesError::InvalidFormat(format!(
                            "restrict_images name {name:?} is not in the source image table"
                        )));
                    };
                    requested[i] = true;
                }
                let mut map = vec![None; n_img];
                let mut next = 0u32;
                for (i, m) in map.iter_mut().enumerate() {
                    if requested[i] {
                        *m = Some(next);
                        next += 1;
                    }
                }
                Some(map)
            }
        };

        let cp = self.cluster_patches.as_ref();
        let accepted: Option<[bool; 256]> = cp.map(|_| {
            let mut mask = [false; 256];
            for &s in &opts.accepted_statuses {
                mask[s as u8 as usize] = true;
            }
            mask
        });

        let starts = clusters.cluster_starts.as_slice().expect("contiguous");
        let member_images = clusters.member_images.as_slice().expect("contiguous");
        let member_features = clusters.member_features.as_slice().expect("contiguous");
        let n_clusters = starts.len() - 1;

        // Pass 1: member selection and cluster survival.
        let mut out_starts: Vec<u32> = Vec::with_capacity(n_clusters + 1);
        out_starts.push(0);
        let mut kept_members: Vec<u32> = Vec::new(); // source member indexes
        let mut new_reference: Vec<u32> = Vec::new(); // per surviving cluster
        let mut span_images: Vec<u32> = Vec::new(); // scratch
        for c in 0..n_clusters {
            let source_ref = cp.map(|cp| cp.reference_members[c]);
            if source_ref == Some(CLUSTER_REFERENCE_UNREFINABLE) {
                continue;
            }
            let (lo, hi) = (starts[c] as usize, starts[c + 1] as usize);
            let sel_start = kept_members.len();
            span_images.clear();
            for m in lo..hi {
                if let Some(mask) = &accepted {
                    if !mask[cp.expect("accepted implies cp").member_status[m] as usize] {
                        continue;
                    }
                }
                let img = member_images[m];
                if let Some(map) = &image_map {
                    if map[img as usize].is_none() {
                        continue;
                    }
                }
                kept_members.push(m as u32);
                if !span_images.contains(&img) {
                    span_images.push(img);
                }
            }
            if (span_images.len() as u32) < opts.min_span {
                kept_members.truncate(sel_start);
                continue;
            }
            // Remap the reference: its new global index when it survived
            // selection, the absent-reference sentinel otherwise.
            let new_ref = match source_ref {
                None => CLUSTER_REFERENCE_UNREFINABLE, // no cluster_patches
                Some(r) => kept_members[sel_start..]
                    .iter()
                    .position(|&m| m == r)
                    .map(|off| (sel_start + off) as u32)
                    .unwrap_or(CLUSTER_REFERENCE_UNREFINABLE),
            };
            if cp.is_some() {
                new_reference.push(new_ref);
            }
            out_starts.push(kept_members.len() as u32);
        }
        let n_out_clusters = out_starts.len() - 1;
        let n_out_members = kept_members.len();

        // Pass 2: gather the member-parallel arrays.
        let out_member_images: Vec<u32> = kept_members
            .iter()
            .map(|&m| {
                let img = member_images[m as usize];
                match &image_map {
                    None => img,
                    Some(map) => map[img as usize].expect("kept member is on a selected image"),
                }
            })
            .collect();
        let out_member_features: Vec<u32> = kept_members
            .iter()
            .map(|&m| member_features[m as usize])
            .collect();

        let out_cluster_patches = cp.map(|cp| {
            let mut affines = Array3::zeros((n_out_members, 2, 3));
            for (k, &m) in kept_members.iter().enumerate() {
                for r in 0..2 {
                    for col in 0..3 {
                        affines[[k, r, col]] = cp.member_affines[[m as usize, r, col]];
                    }
                }
            }
            let gather_f32 = |src: &Array1<f32>| -> Array1<f32> {
                Array1::from_iter(kept_members.iter().map(|&m| src[m as usize]))
            };
            ClusterPatchData {
                reference_members: Array1::from_vec(new_reference.clone()),
                member_status: Array1::from_iter(
                    kept_members.iter().map(|&m| cp.member_status[m as usize]),
                ),
                member_affines: affines,
                member_zncc: gather_f32(&cp.member_zncc),
                member_shift_px: gather_f32(&cp.member_shift_px),
                member_consistency_residual: gather_f32(&cp.member_consistency_residual),
                refine_options: cp.refine_options.clone(),
            }
        });

        // Image table: the requested restriction set (file order) or a copy.
        let keep_image = |i: usize| -> bool {
            match &image_map {
                None => true,
                Some(map) => map[i].is_some(),
            }
        };
        let out_image_names: Vec<String> = self
            .image_names
            .iter()
            .enumerate()
            .filter(|(i, _)| keep_image(*i))
            .map(|(_, n)| n.clone())
            .collect();
        let out_feature_tool_hashes: Vec<[u8; 16]> = self
            .feature_tool_hashes
            .iter()
            .enumerate()
            .filter(|(i, _)| keep_image(*i))
            .map(|(_, h)| *h)
            .collect();
        let out_sift_content_hashes: Vec<[u8; 16]> = self
            .sift_content_hashes
            .iter()
            .enumerate()
            .filter(|(i, _)| keep_image(*i))
            .map(|(_, h)| *h)
            .collect();
        let out_feature_counts: Array1<u32> = Array1::from_iter(
            self.feature_counts
                .iter()
                .enumerate()
                .filter(|(i, _)| keep_image(*i))
                .map(|(_, &v)| v),
        );
        let n_out_images = out_image_names.len();
        let out_image_dims: Option<Array2<u32>> = self.image_dims.as_ref().map(|dims| {
            let mut out = Array2::zeros((n_out_images, 2));
            let mut row = 0;
            for i in 0..n_img {
                if keep_image(i) {
                    out[[row, 0]] = dims[[i, 0]];
                    out[[row, 1]] = dims[[i, 1]];
                    row += 1;
                }
            }
            out
        });

        // Metadata: inherited, with updated counts/flags and the derivation
        // provenance recorded under matching_options["cluster_selection"].
        let mut metadata = self.metadata.clone();
        metadata.version = MATCHES_FORMAT_VERSION;
        metadata.image_count = n_out_images as u32;
        metadata.image_pair_count = None;
        metadata.match_count = None;
        metadata.cluster_count = Some(n_out_clusters as u32);
        metadata.cluster_member_count = Some(n_out_members as u32);
        metadata.has_two_view_geometries = false;
        metadata.has_clusters = true;
        metadata.has_cluster_patches = out_cluster_patches.is_some();
        metadata.matching_options.insert(
            "cluster_selection".into(),
            opts.provenance(&self.content_hash.content_xxh128),
        );

        Ok(MatchesData {
            metadata,
            content_hash: MatchesContentHash {
                metadata_xxh128: String::new(),
                images_xxh128: String::new(),
                image_pairs_xxh128: None,
                clusters_xxh128: None,
                cluster_patches_xxh128: None,
                two_view_geometries_xxh128: None,
                content_xxh128: String::new(),
            },
            image_names: out_image_names,
            feature_tool_hashes: out_feature_tool_hashes,
            sift_content_hashes: out_sift_content_hashes,
            feature_counts: out_feature_counts,
            image_dims: out_image_dims,
            image_pairs: None,
            clusters: Some(ClustersData {
                cluster_starts: Array1::from_vec(out_starts),
                member_images: Array1::from_vec(out_member_images),
                member_features: Array1::from_vec(out_member_features),
                matcher_options: clusters.matcher_options.clone(),
            }),
            cluster_patches: out_cluster_patches,
            two_view_geometries: None,
        })
    }

    /// Per-cluster worst (maximum) finite `member_consistency_residual` over
    /// each cluster's members, as `f64`; `f64::INFINITY` for clusters with
    /// no finite residual. `None` when the file lacks the cluster backbone
    /// or the `cluster_patches/` section.
    pub fn cluster_worst_consistency(&self) -> Option<Array1<f64>> {
        let clusters = self.clusters.as_ref()?;
        let cp = self.cluster_patches.as_ref()?;
        let starts = clusters.cluster_starts.as_slice().expect("contiguous");
        let mut out = Vec::with_capacity(starts.len() - 1);
        for c in 0..starts.len() - 1 {
            let mut worst = f64::INFINITY;
            let mut any = false;
            for m in starts[c] as usize..starts[c + 1] as usize {
                let v = cp.member_consistency_residual[m];
                if v.is_finite() && (!any || f64::from(v) > worst) {
                    worst = f64::from(v);
                    any = true;
                }
            }
            out.push(if any { worst } else { f64::INFINITY });
        }
        Some(Array1::from_vec(out))
    }
}

impl ClusterPatchData {
    /// `(M, 2)` member absolute keypoint positions — the last column of
    /// `member_affines` (`p = A·x_ref + t`; the reference row's own
    /// position for reference members; zeros where not evaluated).
    pub fn member_positions(&self) -> Array2<f64> {
        let m = self.member_affines.shape()[0];
        let mut out = Array2::zeros((m, 2));
        for k in 0..m {
            out[[k, 0]] = self.member_affines[[k, 0, 2]];
            out[[k, 1]] = self.member_affines[[k, 1, 2]];
        }
        out
    }

    /// `(M, 2, 2)` member ← reference patch warps — the leading 2×2 block
    /// of `member_affines` (identity for reference rows; zeros where not
    /// evaluated).
    pub fn member_warps(&self) -> Array3<f64> {
        let m = self.member_affines.shape()[0];
        let mut out = Array3::zeros((m, 2, 2));
        for k in 0..m {
            for r in 0..2 {
                for c in 0..2 {
                    out[[k, r, c]] = self.member_affines[[k, r, c]];
                }
            }
        }
        out
    }

    /// The refinement patch half-width in pixels, normalized across the
    /// `refine_options` key generations: `patch_size` (the full patch edge,
    /// current) divided by 2, or the legacy `radius` (already a half-width)
    /// as-is. `None` when `refine_options` carries neither as a number.
    pub fn refine_radius(&self) -> Option<f64> {
        let opts = self.refine_options.as_object()?;
        if let Some(v) = opts.get("patch_size").and_then(|v| v.as_f64()) {
            return Some(v / 2.0);
        }
        opts.get("radius").and_then(|v| v.as_f64())
    }
}
