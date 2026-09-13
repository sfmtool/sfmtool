// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use crate::{DecodedNode, KdfError, KdfFile, KdfScalar, LazyKdForestOptions, Verification};

/// Read and semantically verify every tree, descriptor, and origin block.
pub fn verify_kdf<S: KdfScalar>(
    path: &Path,
    options: LazyKdForestOptions,
) -> Result<Verification, KdfError> {
    let file = KdfFile::<S>::open(path, options)?;
    let mut reference: Option<Vec<S>> = None;
    let mut chunk_addresses = HashSet::new();
    for ti in 0..file.tree_count() {
        let mut seen_nodes = HashSet::new();
        let mut seen_features = vec![false; file.len()];
        let mut vectors = vec![S::ZERO; file.len() * file.dim()];
        let mut stack = Vec::new();
        if let Some(root) = file.root(ti) {
            stack.push((root, Vec::<Constraint<S>>::new()));
        }
        while let Some((address, constraints)) = stack.pop() {
            if !seen_nodes.insert(address.logical) {
                return Err(KdfError::InvalidFormat(format!(
                    "tree {ti} revisits logical node {}",
                    address.logical
                )));
            }
            chunk_addresses.insert((ti, address.chunk));
            match file.node(ti as u32, address)? {
                DecodedNode::Internal {
                    split_dimension,
                    split,
                    left,
                    right,
                } => {
                    let mut rc = constraints.clone();
                    rc.push(Constraint {
                        dim: split_dimension as usize,
                        split,
                        left: false,
                    });
                    let mut lc = constraints;
                    lc.push(Constraint {
                        dim: split_dimension as usize,
                        split,
                        left: true,
                    });
                    stack.push((right, rc));
                    stack.push((left, lc));
                }
                DecodedNode::Leaf { .. } => {
                    let leaf = file.leaf(ti as u32, address)?;
                    for &id in &leaf.feature_ids {
                        let Some(mark) = seen_features.get_mut(id as usize) else {
                            return Err(KdfError::InvalidFormat(
                                "leaf feature ID out of range".into(),
                            ));
                        };
                        if std::mem::replace(mark, true) {
                            return Err(KdfError::InvalidFormat(format!(
                                "tree {ti} repeats feature ID {id}"
                            )));
                        }
                        let row = file.vector(id)?;
                        for c in &constraints {
                            let ord = row[c.dim].total_cmp(c.split);
                            if (c.left && ord == std::cmp::Ordering::Greater)
                                || (!c.left && ord == std::cmp::Ordering::Less)
                            {
                                return Err(KdfError::InvalidFormat(format!(
                                    "tree {ti} violates a split constraint"
                                )));
                            }
                        }
                        vectors[id as usize * file.dim()..(id as usize + 1) * file.dim()]
                            .copy_from_slice(&row);
                    }
                }
            }
        }
        let declared_nodes: usize = file.metadata().trees[ti]
            .chunks
            .iter()
            .map(|c| c.node_count as usize)
            .sum();
        if seen_nodes.len() != declared_nodes || seen_features.iter().any(|v| !v) {
            return Err(KdfError::InvalidFormat(format!(
                "tree {ti} is incomplete or has unreachable nodes"
            )));
        }
        if let Some(expected) = &reference {
            if expected != &vectors {
                return Err(KdfError::InvalidFormat(format!(
                    "tree {ti} vectors differ from tree 0"
                )));
            }
        } else {
            reference = Some(vectors);
        }
    }
    let origin_blocks = file
        .metadata()
        .origin_block_rows
        .map_or(0, |q| file.len().div_ceil(q as usize));
    if origin_blocks > 0 {
        let ids: Vec<u32> = (0..file.len() as u32).collect();
        let origins = file.resolve_origins(&ids)?.expect("SIFT mode");
        let image_count = file.image_table()?.expect("SIFT mode").names.len();
        let mut seen = HashSet::new();
        for origin in origins {
            if origin.image_index as usize >= image_count || !seen.insert(origin) {
                return Err(KdfError::InvalidFormat(
                    "invalid or duplicate feature origin".into(),
                ));
            }
        }
    }
    let descriptor_blocks = file
        .len()
        .div_ceil(file.metadata().descriptor_block_rows as usize);
    let geometry_blocks = if file.has_feature_geometry() {
        for block in 0..descriptor_blocks {
            file.feature_geometry_block(block as u32)?;
        }
        descriptor_blocks
    } else {
        0
    };
    // Descriptors not reached by malformed incomplete trees are still
    // forced above because every valid tree must cover every feature.
    Ok(Verification {
        trees: file.tree_count(),
        chunks: chunk_addresses.len(),
        descriptor_blocks,
        geometry_blocks,
        origin_blocks,
        features: file.len(),
    })
}

#[derive(Clone)]
struct Constraint<S> {
    dim: usize,
    split: S,
    left: bool,
}

/// Verify all referenced SIFT identities, bounds, and descriptor bytes.
pub fn verify_sift_sources(
    path: &Path,
    options: LazyKdForestOptions,
) -> Result<Verification, KdfError> {
    let file = KdfFile::<u8>::open(path, options)?;
    let Some(table) = file.image_table()? else {
        return Err(KdfError::InvalidFormat("forest has no SIFT sources".into()));
    };
    let workspace =
        resolve_workspace(path, file.metadata().workspace.as_ref().expect("validated"))?;
    let ids: Vec<u32> = (0..file.len() as u32).collect();
    let origins = file.resolve_origins(&ids)?.expect("SIFT mode");
    let mut by_image: HashMap<u32, Vec<(u32, u32)>> = HashMap::new();
    for (id, origin) in origins.iter().enumerate() {
        by_image
            .entry(origin.image_index)
            .or_default()
            .push((id as u32, origin.image_feature_index));
    }
    let prefix = &file
        .metadata()
        .workspace
        .as_ref()
        .expect("validated")
        .contents
        .feature_prefix_dir;
    for (image_index, members) in by_image {
        let image_name = Path::new(&table.names[image_index as usize]);
        let parent = image_name.parent().unwrap_or_else(|| Path::new(""));
        let base = image_name
            .file_name()
            .ok_or_else(|| KdfError::InvalidFormat("image name has no basename".into()))?;
        let mut sift_name = base.to_os_string();
        sift_name.push(".sift");
        let sift_path = workspace.join(parent).join(prefix).join(sift_name);
        if !sift_path.is_file() {
            return Err(KdfError::MissingSource(sift_path));
        }
        let (_, metadata, hashes) = sift_format::read_sift_metadata(&sift_path)?;
        let tool = parse_hash_bytes(&hashes.feature_tool_xxh128)?;
        let content = parse_hash_bytes(&hashes.content_xxh128)?;
        if tool != table.feature_tool_hashes[image_index as usize]
            || content != table.sift_content_hashes[image_index as usize]
        {
            return Err(KdfError::Integrity(format!(
                "SIFT identity mismatch for {}",
                sift_path.display()
            )));
        }
        let sift = sift_format::read_sift(&sift_path)?;
        for (id, image_feature) in members {
            if image_feature >= metadata.feature_count {
                return Err(KdfError::InvalidFormat(
                    "origin image_feature_index out of range".into(),
                ));
            }
            let stored = file.vector_for_verify(id)?;
            if stored
                != sift
                    .descriptors
                    .row(image_feature as usize)
                    .as_slice()
                    .expect("contiguous")
            {
                return Err(KdfError::Integrity(format!(
                    "source descriptor differs for feature {id}"
                )));
            }
            let position = sift.positions_xy.row(image_feature as usize);
            let feature = image_feature as usize;
            let expected = [
                [position[0], position[1]],
                [
                    sift.affine_shapes[[feature, 0, 0]],
                    sift.affine_shapes[[feature, 0, 1]],
                ],
                [
                    sift.affine_shapes[[feature, 1, 0]],
                    sift.affine_shapes[[feature, 1, 1]],
                ],
            ];
            let actual = file.feature_geometry(id)?.expect("SIFT geometry");
            if actual
                .iter()
                .flatten()
                .zip(expected.iter().flatten())
                .any(|(stored, source)| stored.to_bits() != source.to_bits())
            {
                return Err(KdfError::Integrity(format!(
                    "source feature geometry differs for feature {id}"
                )));
            }
        }
    }
    Ok(Verification {
        trees: file.tree_count(),
        chunks: 0,
        descriptor_blocks: file
            .len()
            .div_ceil(file.metadata().descriptor_block_rows as usize),
        geometry_blocks: file
            .len()
            .div_ceil(file.metadata().descriptor_block_rows as usize),
        origin_blocks: file
            .metadata()
            .origin_block_rows
            .map_or(0, |q| file.len().div_ceil(q as usize)),
        features: file.len(),
    })
}

fn resolve_workspace(
    kdf: &Path,
    workspace: &crate::KdfWorkspaceMetadata,
) -> Result<PathBuf, KdfError> {
    let parent = kdf.parent().unwrap_or_else(|| Path::new("."));
    if !workspace.relative_path.is_empty() {
        let p = parent.join(&workspace.relative_path);
        if p.is_dir() {
            return Ok(p);
        }
    }
    if !workspace.absolute_path.is_empty() {
        let p = PathBuf::from(&workspace.absolute_path);
        if p.is_dir() {
            return Ok(p);
        }
    }
    let mut at = parent;
    loop {
        if at.join(".sfm-workspace.json").is_file() {
            return Ok(at.to_path_buf());
        }
        let Some(next) = at.parent() else { break };
        at = next;
    }
    Err(KdfError::MissingSource(parent.to_path_buf()))
}

fn parse_hash_bytes(s: &str) -> Result<[u8; 16], KdfError> {
    let v = u128::from_str_radix(s, 16)
        .map_err(|_| KdfError::InvalidFormat("invalid SIFT hash encoding".into()))?;
    // Hash arrays use the same byte order as Python's `bytes.fromhex` and the
    // other archive formats: the first two hex digits are the first stored
    // byte. The u128 is only a convenient strict parser here.
    Ok(v.to_be_bytes())
}

impl From<sift_format::SiftError> for KdfError {
    fn from(value: sift_format::SiftError) -> Self {
        KdfError::InvalidFormat(format!("SIFT source: {value}"))
    }
}

impl KdfFile<u8> {
    fn vector_for_verify(&self, id: u32) -> Result<Vec<u8>, KdfError> {
        self.vector(id)
    }
}
