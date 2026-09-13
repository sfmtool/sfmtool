// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashSet;
use std::io::{Seek, Write};
use std::path::Path;

use sfmtool_archive_io::{format_hash, write_binary_entry, write_json_entry};
use xxhash_rust::xxh3::{xxh3_128, Xxh3};
use zip::ZipWriter;

use crate::types::*;

struct PackedChunk<S: KdfScalar> {
    logical_nodes: Vec<u32>,
    nodes: Vec<DecodedNode<S>>,
    feature_ids: Vec<u32>,
}

impl<S: KdfScalar> PackedChunk<S> {
    fn decoded_bytes(&self) -> usize {
        NODE_COLUMNS * 4 * self.nodes.len()
            + std::mem::size_of::<S>() * self.nodes.len()
            + 4 * self.feature_ids.len()
    }
}

/// Write one immutable persistent forest. The destination must not exist.
pub fn write_kdf<S: KdfScalar>(
    path: &Path,
    data: &KdfForestData<'_, S>,
    sources: Option<&KdfSiftSources>,
    options: &KdfWriteOptions,
) -> Result<(), KdfError> {
    if path.exists() {
        return Err(KdfError::InvalidFormat(format!(
            "destination already exists: {}",
            path.display()
        )));
    }
    validate_input(data, sources, options)?;
    let packed: Vec<Vec<PackedChunk<S>>> = data
        .trees
        .iter()
        .map(|t| pack_tree(t, options))
        .collect::<Result<_, _>>()?;
    sfmtool_archive_io::write_atomically(path, |file| {
        write_into(file, data, sources, options, &packed)
    })
}

fn validate_input<S: KdfScalar>(
    data: &KdfForestData<'_, S>,
    sources: Option<&KdfSiftSources>,
    options: &KdfWriteOptions,
) -> Result<(), KdfError> {
    if data.dimension == 0 || data.dimension > u16::MAX as usize {
        return Err(KdfError::InvalidFormat(
            "dimension must be in 1..=65535".into(),
        ));
    }
    if data.feature_count > u32::MAX as usize {
        return Err(KdfError::InvalidFormat(
            "feature_count exceeds uint32".into(),
        ));
    }
    let expected = data
        .feature_count
        .checked_mul(data.dimension)
        .ok_or_else(|| KdfError::ResourceLimit("vector shape overflow".into()))?;
    if data.vectors.len() != expected {
        return Err(KdfError::ShapeMismatch(format!(
            "vectors contain {} scalars, expected {expected}",
            data.vectors.len()
        )));
    }
    if data.vectors.iter().any(|&v| !v.is_finite()) {
        return Err(KdfError::InvalidFormat(
            "vectors contain non-finite values".into(),
        ));
    }
    if data.trees.is_empty() {
        return Err(KdfError::InvalidFormat(
            "at least one tree is required".into(),
        ));
    }
    if options.target_chunk_bytes == 0 || options.origin_block_rows == 0 {
        return Err(KdfError::InvalidFormat(
            "chunk and origin block targets must be positive".into(),
        ));
    }
    if options.origin_block_rows > u32::MAX as usize {
        return Err(KdfError::InvalidFormat(
            "origin_block_rows exceeds uint32".into(),
        ));
    }
    if options.target_descriptor_block_bytes == 0 {
        return Err(KdfError::InvalidFormat(
            "descriptor block target must be positive".into(),
        ));
    }
    let row_bytes = data.dimension * std::mem::size_of::<S>();
    if (options.target_descriptor_block_bytes / row_bytes).max(1) > u32::MAX as usize {
        return Err(KdfError::InvalidFormat(
            "descriptor block row count exceeds uint32".into(),
        ));
    }
    for (ti, tree) in data.trees.iter().enumerate() {
        validate_tree(tree, data.feature_count, data.dimension, ti)?;
    }
    if let Some(src) = sources {
        if S::TYPE_NAME != "uint8" || data.dimension != 128 {
            return Err(KdfError::InvalidFormat(
                "SIFT references require unchanged 128-D uint8 descriptors".into(),
            ));
        }
        if src.origins.len() != data.feature_count {
            return Err(KdfError::ShapeMismatch(
                "origins length must equal feature_count".into(),
            ));
        }
        if src.geometry.len() != data.feature_count {
            return Err(KdfError::ShapeMismatch(
                "geometry length must equal feature_count".into(),
            ));
        }
        if src
            .geometry
            .iter()
            .flatten()
            .flatten()
            .any(|v| !v.is_finite())
        {
            return Err(KdfError::InvalidFormat(
                "feature geometry contains non-finite values".into(),
            ));
        }
        if src.image_names.len() != src.feature_tool_hashes.len()
            || src.image_names.len() != src.sift_content_hashes.len()
        {
            return Err(KdfError::ShapeMismatch(
                "image names and hash arrays differ in length".into(),
            ));
        }
        let mut names = HashSet::new();
        let mut pairs = HashSet::new();
        if !src
            .image_names
            .iter()
            .all(|n| valid_relative_posix(n) && names.insert(n))
        {
            return Err(KdfError::InvalidFormat(
                "image names must be unique relative POSIX paths".into(),
            ));
        }
        for &o in &src.origins {
            if o.image_index as usize >= src.image_names.len() {
                return Err(KdfError::InvalidFormat(
                    "origin image index is out of range".into(),
                ));
            }
            if !pairs.insert(o) {
                return Err(KdfError::InvalidFormat(
                    "duplicate image-feature origin pair".into(),
                ));
            }
        }
    }
    Ok(())
}

fn valid_relative_posix(name: &str) -> bool {
    !name.is_empty()
        && !name.starts_with('/')
        && !name.contains('\\')
        && name.split('/').all(|p| !matches!(p, "" | "." | ".."))
}

fn validate_tree<S: KdfScalar>(
    tree: &KdfTree<S>,
    n: usize,
    dim: usize,
    ti: usize,
) -> Result<(), KdfError> {
    if n == 0 {
        if !tree.nodes.is_empty() || !tree.feature_ids.is_empty() {
            return Err(KdfError::InvalidFormat(format!("tree {ti} must be empty")));
        }
        return Ok(());
    }
    if tree.nodes.is_empty() || tree.nodes.len() > u32::MAX as usize || tree.feature_ids.len() != n
    {
        return Err(KdfError::ShapeMismatch(format!(
            "tree {ti} does not cover all features or exceeds uint32 nodes"
        )));
    }
    let mut seen_nodes = vec![false; tree.nodes.len()];
    let mut stack = vec![0u32];
    let mut leaf_ranges = Vec::new();
    while let Some(id) = stack.pop() {
        let Some(mark) = seen_nodes.get_mut(id as usize) else {
            return Err(KdfError::InvalidFormat(format!(
                "tree {ti} child index out of range"
            )));
        };
        if std::mem::replace(mark, true) {
            return Err(KdfError::InvalidFormat(format!(
                "tree {ti} contains a cycle or shared child"
            )));
        }
        match tree.nodes[id as usize] {
            KdfNode::Internal {
                split_dimension,
                split,
                left,
                right,
            } => {
                if split_dimension as usize >= dim || !split.is_finite() {
                    return Err(KdfError::InvalidFormat(format!(
                        "tree {ti} has invalid split"
                    )));
                }
                stack.push(right);
                stack.push(left);
            }
            KdfNode::Leaf { start, len } => {
                let end = (start as usize)
                    .checked_add(len as usize)
                    .ok_or_else(|| KdfError::InvalidFormat("leaf range overflow".into()))?;
                if len == 0 || end > n {
                    return Err(KdfError::InvalidFormat(format!(
                        "tree {ti} has invalid leaf range"
                    )));
                }
                leaf_ranges.push((start as usize, end));
            }
        }
    }
    if seen_nodes.iter().any(|x| !x) {
        return Err(KdfError::InvalidFormat(format!(
            "tree {ti} has unreachable nodes"
        )));
    }
    leaf_ranges.sort_unstable();
    let mut covered = 0;
    for (start, end) in leaf_ranges {
        if start != covered {
            return Err(KdfError::InvalidFormat(format!(
                "tree {ti} leaf ranges overlap or leave a gap"
            )));
        }
        covered = end;
    }
    if covered != n {
        return Err(KdfError::InvalidFormat(format!(
            "tree {ti} leaves do not cover every feature row"
        )));
    }
    let mut ids = tree.feature_ids.clone();
    ids.sort_unstable();
    if ids.iter().copied().ne(0..n as u32) {
        return Err(KdfError::InvalidFormat(format!(
            "tree {ti} feature ids are not a permutation"
        )));
    }
    Ok(())
}

fn subtree_weight<S: KdfScalar>(
    tree: &KdfTree<S>,
    node: u32,
    memo: &mut [usize],
) -> Result<usize, KdfError> {
    if memo[node as usize] != 0 {
        return Ok(memo[node as usize]);
    }
    let base = NODE_COLUMNS * 4 + std::mem::size_of::<S>();
    let w = match tree.nodes[node as usize] {
        KdfNode::Internal { left, right, .. } => {
            let left_weight = subtree_weight(tree, left, memo)?;
            let right_weight = subtree_weight(tree, right, memo)?;
            base.checked_add(left_weight)
                .and_then(|v| v.checked_add(right_weight))
        }
        KdfNode::Leaf { len, .. } => base.checked_add(
            (len as usize)
                .checked_mul(4)
                .ok_or_else(|| KdfError::ResourceLimit("subtree size overflow".into()))?,
        ),
    }
    .ok_or_else(|| KdfError::ResourceLimit("subtree size overflow".into()))?;
    memo[node as usize] = w;
    Ok(w)
}

fn pack_tree<S: KdfScalar>(
    tree: &KdfTree<S>,
    options: &KdfWriteOptions,
) -> Result<Vec<PackedChunk<S>>, KdfError> {
    if tree.nodes.is_empty() {
        return Ok(Vec::new());
    }
    let mut memo = vec![0; tree.nodes.len()];
    subtree_weight(tree, 0, &mut memo)?;
    let mut subtree_roots = Vec::new();
    let mut routing = Vec::new();
    let mut queue = std::collections::VecDeque::from([0u32]);
    while let Some(node) = queue.pop_front() {
        let is_leaf = matches!(tree.nodes[node as usize], KdfNode::Leaf { .. });
        if memo[node as usize] <= options.target_chunk_bytes || is_leaf {
            subtree_roots.push(node);
        } else if let KdfNode::Internal { left, right, .. } = tree.nodes[node as usize] {
            routing.push(node);
            queue.push_back(left);
            queue.push_back(right);
        }
    }
    // Routing nodes use actual node bytes and deterministic breadth-first groups.
    let per_node = NODE_COLUMNS * 4 + std::mem::size_of::<S>();
    let routing_per_chunk = (options.target_chunk_bytes / per_node).max(1);
    let mut chunk_nodes: Vec<Vec<u32>> = routing
        .chunks(routing_per_chunk)
        .map(|v| v.to_vec())
        .collect();
    // Complete subtrees are ordered by source preorder/logical ID.
    subtree_roots.sort_unstable();
    for root in subtree_roots {
        let mut ids = Vec::new();
        collect_preorder(tree, root, &mut ids);
        chunk_nodes.push(ids);
    }
    let mut addresses = vec![(0u32, 0u32); tree.nodes.len()];
    for (ci, nodes) in chunk_nodes.iter().enumerate() {
        for (li, &logical) in nodes.iter().enumerate() {
            addresses[logical as usize] = (ci as u32, li as u32);
        }
    }
    let mut out = Vec::with_capacity(chunk_nodes.len());
    for nodes_in_chunk in chunk_nodes {
        let mut decoded_nodes = Vec::with_capacity(nodes_in_chunk.len());
        let mut feature_ids = Vec::new();
        for &logical in &nodes_in_chunk {
            match tree.nodes[logical as usize] {
                KdfNode::Internal {
                    split_dimension,
                    split,
                    left,
                    right,
                } => {
                    let (lc, ln) = addresses[left as usize];
                    let (rc, rn) = addresses[right as usize];
                    decoded_nodes.push(DecodedNode::Internal {
                        split_dimension,
                        split,
                        left: NodeAddress {
                            chunk: lc,
                            local: ln,
                            logical: left,
                        },
                        right: NodeAddress {
                            chunk: rc,
                            local: rn,
                            logical: right,
                        },
                    });
                }
                KdfNode::Leaf { start, len } => {
                    let local_start = feature_ids.len() as u32;
                    let ids = &tree.feature_ids[start as usize..(start + len) as usize];
                    feature_ids.extend_from_slice(ids);
                    decoded_nodes.push(DecodedNode::Leaf {
                        start: local_start,
                        len,
                    });
                }
            }
        }
        out.push(PackedChunk {
            logical_nodes: nodes_in_chunk,
            nodes: decoded_nodes,
            feature_ids,
        });
    }
    Ok(out)
}

fn collect_preorder<S: KdfScalar>(tree: &KdfTree<S>, node: u32, out: &mut Vec<u32>) {
    out.push(node);
    if let KdfNode::Internal { left, right, .. } = tree.nodes[node as usize] {
        collect_preorder(tree, left, out);
        collect_preorder(tree, right, out);
    }
}

fn write_into<W: Write + Seek, S: KdfScalar>(
    writer: W,
    data: &KdfForestData<'_, S>,
    sources: Option<&KdfSiftSources>,
    options: &KdfWriteOptions,
    packed: &[Vec<PackedChunk<S>>],
) -> Result<(), KdfError> {
    let row_bytes = data
        .dimension
        .checked_mul(std::mem::size_of::<S>())
        .ok_or_else(|| KdfError::ResourceLimit("row byte size overflow".into()))?;
    let descriptor_rows = (options.target_descriptor_block_bytes / row_bytes).max(1);
    let metadata = Metadata {
        format: "kdf".into(),
        version: KDF_FORMAT_VERSION,
        scalar_type: S::TYPE_NAME.into(),
        metric: "squared_l2".into(),
        feature_count: data.feature_count as u32,
        dimension: data.dimension as u16,
        node_kinds: vec!["internal".into(), "leaf".into()],
        target_chunk_bytes: options.target_chunk_bytes as u64,
        trees: packed
            .iter()
            .map(|chunks| TreeMetadata {
                root: (!chunks.is_empty()).then_some([0, 0]),
                chunks: chunks
                    .iter()
                    .map(|c| ChunkMetadata {
                        node_count: c.nodes.len() as u32,
                        feature_count: c.feature_ids.len() as u32,
                        decoded_bytes: c.decoded_bytes() as u64,
                    })
                    .collect(),
            })
            .collect(),
        feature_source: if sources.is_some() {
            "sift_files"
        } else {
            "none"
        }
        .into(),
        descriptor_block_rows: descriptor_rows as u32,
        origin_block_rows: sources.map(|_| options.origin_block_rows as u32),
        workspace: sources.map(|s| s.workspace.clone()),
        provenance: data.provenance.clone(),
    };
    let mut zip = ZipWriter::new(writer);
    let metadata_raw = write_json_entry(
        &mut zip,
        "metadata.json.zst",
        &metadata,
        options.compression_level,
    )?;
    let metadata_digest = xxh3_128(&metadata_raw);
    let mut section_digests = vec![metadata_digest];

    let (images_digest, origin_digests) = if let Some(src) = sources {
        let imeta = ImagesMetadata {
            image_count: src.image_names.len() as u32,
        };
        // Section hash order is lexicographic path order, intentionally distinct
        // from the origin-pair order below.
        let feature_hash_raw = bytemuck::cast_slice(src.feature_tool_hashes.as_slice());
        write_binary_entry(
            &mut zip,
            &format!(
                "images/feature_tool_hashes.{}.uint128.zst",
                src.image_names.len()
            ),
            feature_hash_raw,
            options.compression_level,
        )?;
        let imeta_raw = write_json_entry(
            &mut zip,
            "images/metadata.json.zst",
            &imeta,
            options.compression_level,
        )?;
        let names_raw = write_json_entry(
            &mut zip,
            "images/names.json.zst",
            &src.image_names,
            options.compression_level,
        )?;
        let sift_hash_raw = bytemuck::cast_slice(src.sift_content_hashes.as_slice());
        write_binary_entry(
            &mut zip,
            &format!(
                "images/sift_content_hashes.{}.uint128.zst",
                src.image_names.len()
            ),
            sift_hash_raw,
            options.compression_level,
        )?;
        let mut ih = Xxh3::new();
        ih.update(feature_hash_raw);
        ih.update(&imeta_raw);
        ih.update(&names_raw);
        ih.update(sift_hash_raw);
        let images_digest = ih.digest128();
        let mut ods = Vec::new();
        for (b, rows) in src.origins.chunks(options.origin_block_rows).enumerate() {
            let image_indexes: Vec<u32> = rows.iter().map(|o| o.image_index).collect();
            let feature_indexes: Vec<u32> = rows.iter().map(|o| o.image_feature_index).collect();
            let a = bytemuck::cast_slice(image_indexes.as_slice());
            let f = bytemuck::cast_slice(feature_indexes.as_slice());
            write_binary_entry(
                &mut zip,
                &format!("origins/{b}/image_indexes.{}.uint32.zst", rows.len()),
                a,
                options.compression_level,
            )?;
            write_binary_entry(
                &mut zip,
                &format!(
                    "origins/{b}/image_feature_indexes.{}.uint32.zst",
                    rows.len()
                ),
                f,
                options.compression_level,
            )?;
            let mut h = Xxh3::new();
            h.update(a);
            h.update(f);
            ods.push(h.digest128());
        }
        section_digests.push(images_digest);
        section_digests.extend(ods.iter().copied());
        (Some(images_digest), Some(ods))
    } else {
        (None, None)
    };

    let (storage_digest, descriptor_digests, geometry_digests) = {
        let q = descriptor_rows;
        let tree_zero = &data.trees[0].feature_ids;
        let order: &[u32] = match data.descriptor_order {
            Some(explicit) => {
                if explicit.len() != data.feature_count {
                    return Err(KdfError::ShapeMismatch(format!(
                        "descriptor_order has {} entries, expected {}",
                        explicit.len(),
                        data.feature_count
                    )));
                }
                let mut seen = vec![false; data.feature_count];
                for &id in explicit {
                    let slot = seen.get_mut(id as usize).ok_or_else(|| {
                        KdfError::InvalidFormat("descriptor_order has an out-of-range ID".into())
                    })?;
                    if std::mem::replace(slot, true) {
                        return Err(KdfError::InvalidFormat(
                            "descriptor_order repeats an ID".into(),
                        ));
                    }
                }
                explicit
            }
            None => tree_zero,
        };
        let mut storage_rows = vec![0u32; data.feature_count];
        for (row, &id) in order.iter().enumerate() {
            storage_rows[id as usize] = row as u32;
        }
        let raw = bytemuck::cast_slice(storage_rows.as_slice());
        write_binary_entry(
            &mut zip,
            &format!("features/storage_rows.{}.uint32.zst", data.feature_count),
            raw,
            options.compression_level,
        )?;
        let sd = xxh3_128(raw);
        section_digests.push(sd);
        // One container entry of independent per-block frames, plus the offsets
        // that address them. Each frame is still compressed and hashed exactly
        // as a standalone block entry was, so a block read decodes one frame and
        // the digest list is unchanged; what goes away is one ZIP directory
        // record per block, which at small block sizes is most of the file's
        // entries and most of its open cost.
        let mut ds = Vec::new();
        // Stream frames directly: buffering the container duplicates the entire
        // compressed corpus in memory. ZIP64 permits a stream larger than 4 GiB.
        zip.start_file(
            corpus_entry_name::<S>(data.feature_count, data.dimension),
            zip::write::SimpleFileOptions::default()
                .compression_method(zip::CompressionMethod::Stored)
                .large_file(true),
        )?;
        let mut stored = 0u64;
        let mut offsets = vec![0u64];
        for ids in order.chunks(q) {
            let mut block = Vec::with_capacity(ids.len() * data.dimension);
            for &id in ids {
                let base = id as usize * data.dimension;
                block.extend_from_slice(&data.vectors[base..base + data.dimension]);
            }
            let raw: &[u8] = bytemuck::cast_slice(block.as_slice());
            let frame = zstd::encode_all(raw, options.compression_level)?;
            zip.write_all(&frame)?;
            stored += frame.len() as u64;
            offsets.push(stored);
            let d = xxh3_128(raw);
            ds.push(d);
            section_digests.push(d);
        }
        let offsets_raw: &[u8] = bytemuck::cast_slice(offsets.as_slice());
        write_binary_entry(
            &mut zip,
            &block_offsets_entry_name(offsets.len()),
            offsets_raw,
            options.compression_level,
        )?;
        let geometry_digests = if let Some(src) = sources {
            zip.start_file(
                geometry_entry_name(data.feature_count),
                zip::write::SimpleFileOptions::default()
                    .compression_method(zip::CompressionMethod::Stored)
                    .large_file(true),
            )?;
            let mut gs = Vec::new();
            let mut stored = 0u64;
            let mut offsets = vec![0u64];
            for ids in order.chunks(q) {
                let block: Vec<FeatureGeometry> =
                    ids.iter().map(|&id| src.geometry[id as usize]).collect();
                let raw: &[u8] = bytemuck::cast_slice(block.as_slice());
                let frame = zstd::encode_all(raw, options.compression_level)?;
                zip.write_all(&frame)?;
                stored += frame.len() as u64;
                offsets.push(stored);
                let digest = xxh3_128(raw);
                gs.push(digest);
                section_digests.push(digest);
            }
            let offsets_raw: &[u8] = bytemuck::cast_slice(offsets.as_slice());
            write_binary_entry(
                &mut zip,
                &geometry_block_offsets_entry_name(offsets.len()),
                offsets_raw,
                options.compression_level,
            )?;
            Some(gs)
        } else {
            None
        };
        (sd, ds, geometry_digests)
    };

    let mut chunk_digests = Vec::new();
    for (ti, chunks) in packed.iter().enumerate() {
        let mut td = Vec::new();
        for (ci, chunk) in chunks.iter().enumerate() {
            let mut columns = vec![0u32; NODE_COLUMNS * chunk.nodes.len()];
            let mut splits = vec![S::ZERO; chunk.nodes.len()];
            for (i, (&logical, node)) in chunk.logical_nodes.iter().zip(&chunk.nodes).enumerate() {
                columns[i] = match node {
                    DecodedNode::Internal { .. } => 0,
                    DecodedNode::Leaf { .. } => 1,
                };
                columns[chunk.nodes.len() + i] = logical;
                match *node {
                    DecodedNode::Internal {
                        split_dimension,
                        split,
                        left,
                        right,
                    } => {
                        columns[2 * chunk.nodes.len() + i] = split_dimension as u32;
                        columns[3 * chunk.nodes.len() + i] = left.chunk;
                        columns[4 * chunk.nodes.len() + i] = left.local;
                        columns[5 * chunk.nodes.len() + i] = left.logical;
                        columns[6 * chunk.nodes.len() + i] = right.chunk;
                        columns[7 * chunk.nodes.len() + i] = right.local;
                        columns[8 * chunk.nodes.len() + i] = right.logical;
                        splits[i] = split;
                    }
                    DecodedNode::Leaf { start, .. } => columns[9 * chunk.nodes.len() + i] = start,
                }
            }
            // The three integer arrays share one entry: they are always read
            // together — decoding a node needs the columns and the splits, and
            // reaching a leaf needs the IDs — and they compress alike. Vectors
            // stay separate despite also always being read with them, because
            // bulk descriptor bytes in the same zstd frame as these columns make
            // both compress worse.
            let nb: &[u8] = bytemuck::cast_slice(columns.as_slice());
            let sb: &[u8] = bytemuck::cast_slice(splits.as_slice());
            let fb: &[u8] = bytemuck::cast_slice(chunk.feature_ids.as_slice());
            let mut payload = Vec::with_capacity(nb.len() + sb.len() + fb.len());
            payload.extend_from_slice(nb);
            payload.extend_from_slice(sb);
            payload.extend_from_slice(fb);
            write_binary_entry(
                &mut zip,
                &chunk_entry_name::<S>(ti, ci, chunk.nodes.len(), chunk.feature_ids.len()),
                &payload,
                options.compression_level,
            )?;
            let d = xxh3_128(&payload);
            td.push(d);
            section_digests.push(d);
        }
        chunk_digests.push(td);
    }
    let mut whole = Vec::with_capacity(section_digests.len() * 16);
    for d in &section_digests {
        whole.extend_from_slice(&d.to_be_bytes());
    }
    let hashes = ContentHash {
        metadata_xxh128: format_hash(metadata_digest),
        chunks_xxh128: chunk_digests
            .iter()
            .map(|v| v.iter().map(|&d| format_hash(d)).collect())
            .collect(),
        content_xxh128: format_hash(xxh3_128(&whole)),
        storage_rows_xxh128: format_hash(storage_digest),
        descriptor_blocks_xxh128: descriptor_digests.into_iter().map(format_hash).collect(),
        geometry_blocks_xxh128: geometry_digests.map(|v| v.into_iter().map(format_hash).collect()),
        images_xxh128: images_digest.map(format_hash),
        origins_xxh128: origin_digests.map(|v| v.into_iter().map(format_hash).collect()),
    };
    write_json_entry(
        &mut zip,
        "content_hash.json.zst",
        &hashes,
        options.compression_level,
    )?;
    zip.finish()?;
    Ok(())
}
