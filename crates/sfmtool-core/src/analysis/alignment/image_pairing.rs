// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Which image of one reconstruction is which image of another.
//!
//! Aligning two reconstructions starts by pairing their images. A name (the
//! image's path relative to its workspace) is not enough to do that: a folder
//! copied into another workspace layout, or frames renamed after extraction,
//! give the same photograph another name, and two videos' frames in two
//! workspaces give different photographs the same name. The hash of the image
//! file says which photograph an image is, so [`pair_images`] pairs by it and
//! uses the name only where a hash is not known. The rule is stated in full on
//! [`pair_images`] and in `specs/core/analysis/reconstruction-alignment.md`,
//! "Pairing the images".
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
    /// How many of `pairs` were paired by name, where at least one of the two
    /// images has no known file hash.
    pub by_name: usize,
    /// `(source index, target index)` of each pair of images with the same name
    /// whose file hashes are both known and differ: different photographs that
    /// pairing by name alone would have paired. Each name is reported once,
    /// for its first image on each side, ascending by source index. Neither
    /// image is paired with the other, though either may be paired with
    /// another image by its hash.
    pub same_name_different_file: Vec<(u32, u32)>,
}

impl ImagePairing {
    /// How the images were paired, for a report: `images paired: 15 by file
    /// hash, 2 by name`, followed by `; 3 names shared by different files` when
    /// `same_name_different_file` is not empty.
    pub fn summary(&self) -> String {
        let mut text = format!(
            "images paired: {} by file hash, {} by name",
            self.by_file_hash, self.by_name
        );
        match self.same_name_different_file.len() {
            0 => {}
            1 => text.push_str("; 1 name shared by different files"),
            n => text.push_str(&format!("; {n} names shared by different files")),
        }
        text
    }
}

/// Pair the images of `source` with those of `target`.
///
/// The rule, in order:
///
/// 1. **By file hash.** Two images whose file hashes are both known pair only
///    when the hashes are equal, whatever their names. A hash held by exactly
///    one image on each side pairs those two. A hash held by more than one
///    image on a side (the same file under two names) pairs, among the images
///    holding it, only images of the same name; an image of such a group with
///    no same-named partner in the other side's group is left unpaired by
///    hash. These pairs count as `by_file_hash`.
/// 2. **By name.** Each image still unpaired is paired with the image of the
///    same name on the other side that is still unpaired, as long as at least
///    one of the two has no known file hash. These count as `by_name`.
/// 3. **A repeated name** pairs once: in step 2, and among the images of one
///    hash in step 1, the first image of a name on each side is the one that
///    pairs, and a later image of the same name is left unpaired.
///
/// Alongside the pairs, `same_name_different_file` reports the names that
/// pairing by name alone would have paired across two different files.
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

    // ── 2. By name, where a hash is not known ───────────────────────────
    let unpaired_targets = (0..target.len()).filter(|&t| !target_paired[t]);
    let target_by_name = first_by_name(target, unpaired_targets);
    let mut seen = HashSet::new();
    for (s, image) in source.iter().enumerate() {
        if source_paired[s] || !seen.insert(image.name) {
            continue;
        }
        let Some(&t) = target_by_name.get(image.name) else {
            continue;
        };
        if image.file_hash.is_some() && target[t].file_hash.is_some() {
            continue; // both known and, being unpaired, not the same file
        }
        pairs.push((s as u32, t as u32));
        source_paired[s] = true;
        target_paired[t] = true;
    }
    let by_name = pairs.len() - by_file_hash;
    pairs.sort_unstable();

    // ── The pairs a name alone would have made across two files ─────────
    let target_by_name = first_by_name(target, 0..target.len());
    let mut seen = HashSet::new();
    let mut same_name_different_file = Vec::new();
    for (s, image) in source.iter().enumerate() {
        if !seen.insert(image.name) {
            continue;
        }
        let Some(&t) = target_by_name.get(image.name) else {
            continue;
        };
        if let (Some(a), Some(b)) = (image.file_hash, target[t].file_hash) {
            if a != b {
                same_name_different_file.push((s as u32, t as u32));
            }
        }
    }

    ImagePairing {
        pairs,
        by_file_hash,
        by_name,
        same_name_different_file,
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
