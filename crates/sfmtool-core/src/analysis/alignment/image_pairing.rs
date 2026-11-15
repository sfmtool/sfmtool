// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Which image of one reconstruction is which image of another.
//!
//! Aligning two reconstructions starts by pairing their images. A name (the
//! image's path relative to its workspace) misses some pairs: a folder copied
//! into another workspace layout, or frames renamed after extraction, give the
//! same photograph another name. The hash of the image file finds those, so
//! [`pair_images`] pairs by it first and then by name, as a pairing by name
//! alone would. The name pairs whatever the hashes say, so two reconstructions
//! of the same photographs at another resolution or re-encoded still pair. The
//! rule is stated in full on [`pair_images`] and in
//! `specs/core/analysis/reconstruction-alignment.md`, "Pairing the images".
//!
//! [`pair_reconstruction_images`] reads each reconstruction's hashes
//! ([`SfmrReconstruction::find_image_file_hashes`]) and pairs them;
//! [`align_reconstructions`](super::align_reconstructions) and `sfm align`
//! (through the `sfmtool.analysis.pair_images` binding) both pair this way.

use std::collections::hash_map::Entry;
use std::collections::{HashMap, HashSet};

use crate::progress::{Cancelled, Progress};
use crate::SfmrReconstruction;

/// What [`pair_images`] knows of one image: its name and, when known, the
/// hash of its file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImageIdentity<'a> {
    /// The image's name, its path relative to its workspace.
    pub name: &'a str,
    /// The XXH128 of the image file's bytes, or `None` when it is not known.
    pub file_hash: Option<[u8; 16]>,
}

/// The image pairs [`pair_images`] found, and how it found them.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ImagePairing {
    /// `(source index, target index)` of each paired image, ascending by source
    /// index. No index appears twice on either side.
    pub pairs: Vec<(u32, u32)>,
    /// How many of `pairs` were paired by file hash.
    pub by_file_hash: usize,
    /// How many of `pairs` were paired by name.
    pub by_name: usize,
    /// `(source index, target index)` of each pair made by name whose two file
    /// hashes are both known and differ: the same name on different files,
    /// such as the same photograph at another resolution, or the frames of
    /// two videos. A subset of `pairs`, ascending by source index. It is
    /// information for a report and changes no pair.
    pub by_name_different_files: Vec<(u32, u32)>,
}

impl ImagePairing {
    /// How the images were paired, for a report: `images paired: 15 by file
    /// hash, 2 by name`, followed by `, 2 of them different files` when
    /// `by_name_different_files` is not empty.
    pub fn summary(&self) -> String {
        let mut text = format!(
            "images paired: {} by file hash, {} by name",
            self.by_file_hash, self.by_name
        );
        if !self.by_name_different_files.is_empty() {
            text.push_str(&format!(
                ", {} of them different files",
                self.by_name_different_files.len()
            ));
        }
        text
    }
}

/// Pair the images of `source` with those of `target`.
///
/// The rule, in order:
///
/// 1. **By file hash.** Images whose file hashes are known and equal pair,
///    whatever their names. A hash held by exactly one image on each side
///    pairs those two. A hash held by more than one image on a side (the same
///    file under two names) pairs, among the images holding it, only images
///    of the same name, the first image of a name on each side; an image of
///    such a group with no same-named partner in the other side's group is
///    left to step 2. These pairs count as `by_file_hash`.
/// 2. **By name.** Each image still unpaired is paired with the image of the
///    same name on the other side that is still unpaired, whether or not their
///    hashes are known or equal. A repeated name pairs once, its first
///    unpaired image on each side. These pairs count as `by_name`, and those
///    whose two hashes are known and differ are listed in
///    `by_name_different_files`.
///
/// Every pair that pairing by name alone (the first image of a name on each
/// side) makes is therefore still made, unless one of its two images was
/// paired in step 1 with another image of the same file.
///
/// The call reads nothing and runs in time linear in the image counts.
pub fn pair_images(source: &[ImageIdentity<'_>], target: &[ImageIdentity<'_>]) -> ImagePairing {
    let mut source_paired = vec![false; source.len()];
    let mut target_paired = vec![false; target.len()];
    let mut pairs: Vec<(u32, u32)> = Vec::new();

    // ── 1. By file hash ─────────────────────────────────────────────────
    let source_groups = groups_by_hash(source);
    let target_groups: HashMap<[u8; 16], Vec<usize>> = groups_by_hash(target).into_iter().collect();
    for (hash, source_group) in &source_groups {
        let Some(target_group) = target_groups.get(hash) else {
            continue;
        };
        if let ([s], [t]) = (source_group.as_slice(), target_group.as_slice()) {
            pairs.push((*s as u32, *t as u32));
            source_paired[*s] = true;
            target_paired[*t] = true;
            continue;
        }
        // The same file under several names on a side: only the names say
        // which of its images is which.
        let target_by_name = first_by_name(target, target_group.iter().copied());
        let mut seen = HashSet::new();
        for &s in source_group {
            if !seen.insert(source[s].name) {
                continue;
            }
            if let Some(&t) = target_by_name.get(source[s].name) {
                pairs.push((s as u32, t as u32));
                source_paired[s] = true;
                target_paired[t] = true;
            }
        }
    }
    let by_file_hash = pairs.len();

    // ── 2. By name ──────────────────────────────────────────────────────
    let unpaired_targets = (0..target.len()).filter(|&t| !target_paired[t]);
    let target_by_name = first_by_name(target, unpaired_targets);
    let mut seen = HashSet::new();
    let mut by_name_different_files = Vec::new();
    for (s, image) in source.iter().enumerate() {
        if source_paired[s] || !seen.insert(image.name) {
            continue;
        }
        let Some(&t) = target_by_name.get(image.name) else {
            continue;
        };
        pairs.push((s as u32, t as u32));
        source_paired[s] = true;
        target_paired[t] = true;
        if let (Some(a), Some(b)) = (image.file_hash, target[t].file_hash) {
            if a != b {
                by_name_different_files.push((s as u32, t as u32));
            }
        }
    }
    let by_name = pairs.len() - by_file_hash;
    pairs.sort_unstable();

    ImagePairing {
        pairs,
        by_file_hash,
        by_name,
        by_name_different_files,
    }
}

/// Every known file hash and the images holding it, in image order, iterated
/// in the order each hash first appears so the pairs come out the same way on
/// every run.
fn groups_by_hash(images: &[ImageIdentity<'_>]) -> Vec<([u8; 16], Vec<usize>)> {
    let mut slot: HashMap<[u8; 16], usize> = HashMap::new();
    let mut groups: Vec<([u8; 16], Vec<usize>)> = Vec::new();
    for (i, image) in images.iter().enumerate() {
        let Some(hash) = image.file_hash else {
            continue;
        };
        match slot.entry(hash) {
            Entry::Occupied(e) => groups[*e.get()].1.push(i),
            Entry::Vacant(e) => {
                e.insert(groups.len());
                groups.push((hash, vec![i]));
            }
        }
    }
    groups
}

/// `name → index` over `indices`, the first index of each name winning.
fn first_by_name<'a>(
    images: &[ImageIdentity<'a>],
    indices: impl Iterator<Item = usize>,
) -> HashMap<&'a str, usize> {
    let mut map = HashMap::new();
    for i in indices {
        map.entry(images[i].name).or_insert(i);
    }
    map
}

/// The identities of `recon`'s images under `hashes`, one per image.
fn identities<'a>(
    recon: &'a SfmrReconstruction,
    hashes: &[Option<[u8; 16]>],
) -> Vec<ImageIdentity<'a>> {
    recon
        .image_set
        .images
        .iter()
        .zip(hashes)
        .map(|(image, &file_hash)| ImageIdentity {
            name: image.name.as_str(),
            file_hash,
        })
        .collect()
}

/// Pair the images of two reconstructions by [`pair_images`], with each
/// image's file hash read by [`SfmrReconstruction::find_image_file_hashes`].
///
/// A `sift_files` reconstruction's hashes are read from its `.sift` files'
/// metadata, one file per image, so `progress` counts those reads and can
/// cancel between them.
pub fn pair_reconstruction_images(
    source: &SfmrReconstruction,
    target: &SfmrReconstruction,
    progress: &Progress,
) -> Result<ImagePairing, Cancelled> {
    let [source_progress, target_progress] = progress.split([0.5, 0.5]);
    let source_hashes = source.find_image_file_hashes(&source_progress)?;
    let target_hashes = target.find_image_file_hashes(&target_progress)?;
    Ok(pair_images(
        &identities(source, &source_hashes),
        &identities(target, &target_hashes),
    ))
}

#[cfg(test)]
mod tests;
