// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Archive entry names — the single source of truth for `.matches` ZIP paths.
//!
//! Every entry in a `.matches` archive encodes its shape and dtype in its own
//! name (`image_pairs/match_counts.{pair_count}.uint32.zst`), so the name is not
//! decoration: [`read`](crate::read) looks entries up by exact name,
//! [`write`](crate::write) creates them, and [`verify`](crate::verify) rehashes
//! them in lexicographic order. A name written three times is a name that can
//! disagree three ways, and a mismatch surfaces only as a failed lookup or a
//! wrong section hash.
//!
//! One function per entry, so the template exists once. Callers pass the
//! dimension tokens; the function owns the layout.
//!
//! Dimension parameters are `impl Display` rather than `usize` because the same
//! conceptual count reaches these functions in different integer types — a
//! `usize` local in [`read`](crate::read) and [`write`](crate::write), a `u32`
//! deserialized from section metadata in [`verify`](crate::verify). The token is
//! only ever interpolated, so widening the parameter is preferable to `as usize`
//! casts at the call sites. The trade-off is that `Display` is wider than the
//! intent — it would also accept a float or a signed value — so it documents
//! less than a concrete type would.
//!
//! The one exception is [`clusters_cluster_starts`], which takes a concrete
//! `usize` because it does arithmetic on the count rather than only
//! interpolating it.
//!
//! Unlike `.sfmr`, no `.matches` entry has been renamed across format versions,
//! so every function here has a single spelling.
//!
//! ## How the tests use this module
//!
//! `tests::entry_names_are_pinned` calls these functions and compares each
//! result against a literal spelled out in `tests.rs`. Everywhere else, tests
//! that name an archive entry spell it literally rather than calling in here —
//! routing them through this module would make them agree with whatever these
//! functions produce, which is the property under test.
//!
//! That pin covers spelling and shape only. Which count a given entry is sized
//! by is decided at the call sites, not here, so a caller passing
//! `member_count` where `cluster_count` belongs is invisible to it;
//! `tests::archive_entry_names_pin_call_sites` pins the written archives'
//! listings to cover that.

/// `metadata.json.zst` — the top-level `.matches` metadata.
///
/// Top-level entries carry no section prefix and are not part of any section
/// hash; they sit at the archive root alongside the section directories.
pub(crate) fn metadata() -> &'static str {
    "metadata.json.zst"
}

/// `content_hash.json.zst` — the top-level per-section digest record.
pub(crate) fn content_hash() -> &'static str {
    "content_hash.json.zst"
}

/// `images/metadata.json.zst` — image-section metadata.
pub(crate) fn images_metadata() -> &'static str {
    "images/metadata.json.zst"
}

/// `images/names.json.zst` — image file names.
pub(crate) fn images_names() -> &'static str {
    "images/names.json.zst"
}

/// `images/feature_counts` — feature count per image.
pub(crate) fn images_feature_counts(image_count: impl std::fmt::Display) -> String {
    format!("images/feature_counts.{image_count}.uint32.zst")
}

/// `images/image_dims` — `(width, height)` per image.
pub(crate) fn images_image_dims(image_count: impl std::fmt::Display) -> String {
    format!("{}{image_count}.2.uint32.zst", images_image_dims_prefix())
}

/// The `images/image_dims` name up to and including the dot before its count.
///
/// `verify` tests for this entry's *presence* — it is mandatory from version 4
/// and forbidden before it — without knowing `image_count`, so it matches on
/// this prefix. Deriving it here rather than spelling it out there means a
/// rename cannot silently turn that check into a no-op, which would report
/// every valid version 4+ file as missing the entry.
pub(crate) fn images_image_dims_prefix() -> &'static str {
    "images/image_dims."
}

/// `images/feature_tool_hashes` — XXH128 identifying the feature tool per image.
pub(crate) fn images_feature_tool_hashes(image_count: impl std::fmt::Display) -> String {
    format!("images/feature_tool_hashes.{image_count}.uint128.zst")
}

/// `images/sift_content_hashes` — XXH128 of each linked `.sift` file.
pub(crate) fn images_sift_content_hashes(image_count: impl std::fmt::Display) -> String {
    format!("images/sift_content_hashes.{image_count}.uint128.zst")
}

/// `image_pairs/metadata.json.zst` — pairwise-backbone metadata.
pub(crate) fn image_pairs_metadata() -> &'static str {
    "image_pairs/metadata.json.zst"
}

/// `image_pairs/image_index_pairs` — the `(a, b)` image indexes per pair.
pub(crate) fn image_pairs_image_index_pairs(pair_count: impl std::fmt::Display) -> String {
    format!("image_pairs/image_index_pairs.{pair_count}.2.uint32.zst")
}

/// `image_pairs/match_counts` — match count per pair.
pub(crate) fn image_pairs_match_counts(pair_count: impl std::fmt::Display) -> String {
    format!("image_pairs/match_counts.{pair_count}.uint32.zst")
}

/// `image_pairs/match_feature_indexes` — the matched feature index pair.
///
/// Sized by `match_count` (the total across all pairs), not `pair_count`.
pub(crate) fn image_pairs_match_feature_indexes(match_count: impl std::fmt::Display) -> String {
    format!("image_pairs/match_feature_indexes.{match_count}.2.uint32.zst")
}

/// `image_pairs/match_descriptor_distances` — descriptor distance per match.
///
/// Sized by `match_count`, not `pair_count`.
pub(crate) fn image_pairs_match_descriptor_distances(
    match_count: impl std::fmt::Display,
) -> String {
    format!("image_pairs/match_descriptor_distances.{match_count}.float32.zst")
}

/// `two_view_geometries/metadata.json.zst` — two-view-geometry metadata.
pub(crate) fn two_view_geometries_metadata() -> &'static str {
    "two_view_geometries/metadata.json.zst"
}

/// `two_view_geometries/config_types.json.zst` — the config-type name table.
pub(crate) fn two_view_geometries_config_types() -> &'static str {
    "two_view_geometries/config_types.json.zst"
}

/// `two_view_geometries/config_indexes` — config-type index per pair.
pub(crate) fn two_view_geometries_config_indexes(pair_count: impl std::fmt::Display) -> String {
    format!("two_view_geometries/config_indexes.{pair_count}.uint8.zst")
}

/// `two_view_geometries/e_matrices` — essential matrix per pair.
pub(crate) fn two_view_geometries_e_matrices(pair_count: impl std::fmt::Display) -> String {
    format!("two_view_geometries/e_matrices.{pair_count}.3.3.float64.zst")
}

/// `two_view_geometries/f_matrices` — fundamental matrix per pair.
pub(crate) fn two_view_geometries_f_matrices(pair_count: impl std::fmt::Display) -> String {
    format!("two_view_geometries/f_matrices.{pair_count}.3.3.float64.zst")
}

/// `two_view_geometries/h_matrices` — homography per pair.
pub(crate) fn two_view_geometries_h_matrices(pair_count: impl std::fmt::Display) -> String {
    format!("two_view_geometries/h_matrices.{pair_count}.3.3.float64.zst")
}

/// `two_view_geometries/quaternions_wxyz` — relative rotation per pair.
pub(crate) fn two_view_geometries_quaternions_wxyz(pair_count: impl std::fmt::Display) -> String {
    format!("two_view_geometries/quaternions_wxyz.{pair_count}.4.float64.zst")
}

/// `two_view_geometries/translations_xyz` — relative translation per pair.
pub(crate) fn two_view_geometries_translations_xyz(pair_count: impl std::fmt::Display) -> String {
    format!("two_view_geometries/translations_xyz.{pair_count}.3.float64.zst")
}

/// `two_view_geometries/inlier_counts` — inlier count per pair.
pub(crate) fn two_view_geometries_inlier_counts(pair_count: impl std::fmt::Display) -> String {
    format!("two_view_geometries/inlier_counts.{pair_count}.uint32.zst")
}

/// `two_view_geometries/inlier_feature_indexes` — the inlier feature index pair.
///
/// Sized by `inlier_count` (the total across all pairs), not `pair_count`.
pub(crate) fn two_view_geometries_inlier_feature_indexes(
    inlier_count: impl std::fmt::Display,
) -> String {
    format!("two_view_geometries/inlier_feature_indexes.{inlier_count}.2.uint32.zst")
}

/// `clusters/metadata.json.zst` — cluster-backbone metadata.
pub(crate) fn clusters_metadata() -> &'static str {
    "clusters/metadata.json.zst"
}

/// `clusters/cluster_starts` — CSR offsets into the member arrays.
///
/// Takes `cluster_count` and sizes the entry `cluster_count + 1`: the array
/// holds one offset per cluster plus the terminating total, so the `+ 1` is a
/// property of the entry rather than something each caller should re-derive.
pub(crate) fn clusters_cluster_starts(cluster_count: usize) -> String {
    format!("clusters/cluster_starts.{}.uint32.zst", cluster_count + 1)
}

/// `clusters/member_images` — image index per cluster member.
pub(crate) fn clusters_member_images(member_count: impl std::fmt::Display) -> String {
    format!("clusters/member_images.{member_count}.uint32.zst")
}

/// `clusters/member_features` — feature index per cluster member.
pub(crate) fn clusters_member_features(member_count: impl std::fmt::Display) -> String {
    format!("clusters/member_features.{member_count}.uint32.zst")
}

/// `cluster_patches/metadata.json.zst` — cluster-patch metadata.
pub(crate) fn cluster_patches_metadata() -> &'static str {
    "cluster_patches/metadata.json.zst"
}

/// `cluster_patches/reference_members` — the reference member per cluster.
///
/// Sized by `cluster_count`, not `member_count` — the odd one out in this
/// section.
pub(crate) fn cluster_patches_reference_members(cluster_count: impl std::fmt::Display) -> String {
    format!("cluster_patches/reference_members.{cluster_count}.uint32.zst")
}

/// `cluster_patches/member_status` — per-member refinement status code.
pub(crate) fn cluster_patches_member_status(member_count: impl std::fmt::Display) -> String {
    format!("cluster_patches/member_status.{member_count}.uint8.zst")
}

/// `cluster_patches/member_affines` — the 2×3 affine per member.
pub(crate) fn cluster_patches_member_affines(member_count: impl std::fmt::Display) -> String {
    format!("cluster_patches/member_affines.{member_count}.2.3.float64.zst")
}

/// `cluster_patches/member_zncc` — ZNCC against the reference per member.
pub(crate) fn cluster_patches_member_zncc(member_count: impl std::fmt::Display) -> String {
    format!("cluster_patches/member_zncc.{member_count}.float32.zst")
}

/// `cluster_patches/member_shift_px` — refinement shift in pixels per member.
pub(crate) fn cluster_patches_member_shift_px(member_count: impl std::fmt::Display) -> String {
    format!("cluster_patches/member_shift_px.{member_count}.float32.zst")
}

/// `cluster_patches/member_consistency_residual` — consistency residual per member.
pub(crate) fn cluster_patches_member_consistency_residual(
    member_count: impl std::fmt::Display,
) -> String {
    format!("cluster_patches/member_consistency_residual.{member_count}.float32.zst")
}
