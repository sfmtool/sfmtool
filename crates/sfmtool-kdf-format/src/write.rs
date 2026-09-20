// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashSet;
use std::io::{Seek, Write};
use std::path::Path;

use rayon::prelude::*;
use sfmtool_archive_io::{format_hash, write_binary_entry, write_json_entry, SectionDigests};
use sfmtool_progress::Progress;
use xxhash_rust::xxh3::{xxh3_128, Xxh3};
use zip::ZipWriter;

use crate::types::*;

/// How the write's wall time divides, as the shares
/// [`write_kdf`] gives its six stages: validation, tree packing, the
/// heading (metadata, image table, origins and the corpus row map), the
/// descriptor block loop, the geometry block loop, and the packed tree chunks
/// the content hash closes.
///
/// Measured over a 3.0 M-descriptor, 128-D corpus of four trees with SIFT
/// sources at the default options, which is the shape the viewer's SIFT index
/// build writes: 2%, 2%, 2%, 48%, 36%, 2%, and a 9% unreported tail the last
/// share absorbs so the bar does not reach its end and stop there.
///
/// The two block loops are almost the whole write, and they are close to each
/// other despite the descriptor corpus being five times the bytes: the geometry
/// rows are float32 and compress far less well than descriptor bytes do, so zstd
/// spends comparably long on both. Everything else has become small. Validation,
/// the tree packing, the heading and the chunk loop each divide their work
/// across the thread pool or have little to do, and the heading no longer
/// carries an integrity directory that grew with the block count.
///
/// A corpus written without [`KdfSiftSources`] carries no geometry and no
/// origins, and its bar then stands still for those shares rather than
/// reporting wrongly: the weights are a constant estimate of where the time
/// goes, and one that is wrong makes the bar uneven rather than untrue.
const STAGE_SHARES: [f32; 6] = [0.02, 0.02, 0.02, 0.48, 0.36, 0.10];

/// How many blocks one batch of a block loop compresses together.
///
/// The batch is the unit of everything that is not the compression itself: one
/// progress report, one cancellation check, and one ordered hand-off to the
/// serial writer. Two hundred reports over a stage is more than a bar can show,
/// so a batch is a two-hundredth of the stage — until that would hold more than
/// a few thousand blocks, where the cap takes over and a large corpus simply
/// gets more reports than two hundred. The cap is what bounds the extra memory:
/// a batch holds its blocks' raw and compressed bytes at once, which at a
/// default two-KiB block is about sixteen MiB and does not grow with the corpus.
/// The floor keeps a batch worth handing to a thread pool at all.
fn block_batch(blocks: usize) -> usize {
    (blocks / 200).clamp(256, 4096)
}

/// Compress and hash one batch of blocks, in parallel, into writing order.
///
/// `gather` fills a worker-owned buffer with one block's raw bytes; the worker
/// then compresses that buffer into its own frame and digests it. Every frame is
/// independent, so the only thing the caller must still do in order is write
/// them — which is why this returns a `Vec` in batch order rather than writing
/// anything itself.
///
/// The zstd context is per worker and not per block. A block is a couple of KiB
/// and a capture has hundreds of thousands of them, so a context apiece spends
/// the write allocating and clearing match tables sized for a stream of unknown
/// length; that was a measured ten-fold regression. A single-shot compression
/// does not carry state between calls, so a frame is the same bytes whichever
/// worker produced it and however the blocks were batched.
fn encode_batch<F>(
    batch: &[&[u32]],
    level: i32,
    gather: F,
) -> Result<Vec<(Vec<u8>, u128)>, KdfError>
where
    F: Fn(&[u32], &mut Vec<u8>) + Sync,
{
    batch
        .par_iter()
        .map_init(
            || (None::<zstd::bulk::Compressor<'static>>, Vec::<u8>::new()),
            |(slot, raw), ids| -> Result<(Vec<u8>, u128), KdfError> {
                if slot.is_none() {
                    *slot = Some(zstd::bulk::Compressor::new(level)?);
                }
                raw.clear();
                gather(ids, raw);
                let frame = slot.as_mut().expect("initialized").compress(raw)?;
                Ok((frame, xxh3_128(raw)))
            },
        )
        .collect()
}

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

/// Write one immutable persistent forest, saying where it has got to and
/// stopping when it is asked to.
///
/// The destination must not exist unless
/// [`KdfWriteOptions::replace_existing`] says it may be replaced.
///
/// A corpus of a few million descriptors is several hundred megabytes through
/// zstd, which is seconds of work and long enough that a caller drawing a bar
/// needs to hear from it, so the `progress` is a parameter rather than an
/// entry point of its own; a caller with nothing to report through passes
/// [`Progress::none`]. What it reports is a fraction of its own range,
/// weighted by where a write's time goes rather than by how many bytes are
/// behind it (`STAGE_SHARES`); what it hears back is
/// [`Progress::is_cancelled`], read between batches of blocks rather than per
/// block.
///
/// A cancelled write is [`KdfError::Cancelled`] and **leaves `path` as it
/// found it**: nothing where there was nothing, and the file that was there
/// when one was. The whole archive is streamed into a temporary sibling and
/// renamed over `path` only once it is complete, and the sibling is removed on
/// every other exit.
///
/// ```no_run
/// use sfmtool_kdf_format::{write_kdf, KdfForestData, KdfWriteOptions};
/// use sfmtool_progress::Progress;
///
/// # fn example(data: &KdfForestData<'_, u8>) -> Result<(), sfmtool_kdf_format::KdfError> {
/// write_kdf(
///     "corpus.kdf".as_ref(),
///     data,
///     None,
///     &KdfWriteOptions::default(),
///     &Progress::none(),
/// )
/// # }
/// ```
pub fn write_kdf<S: KdfScalar>(
    path: &Path,
    data: &KdfForestData<'_, S>,
    sources: Option<&KdfSiftSources>,
    options: &KdfWriteOptions,
    progress: &Progress<'_>,
) -> Result<(), KdfError> {
    if !options.replace_existing && path.exists() {
        return Err(KdfError::InvalidFormat(format!(
            "destination already exists: {}",
            path.display()
        )));
    }
    let [checking, packing, heading, descriptors, geometry, chunks] = progress.split(STAGE_SHARES);
    {
        let _phase = checking.detail_phase("validate");
        validate_input(data, sources, options)?;
    }
    checking.set_fraction(1.0);
    checking.check_cancel()?;
    let packed: Vec<Vec<PackedChunk<S>>> = {
        let _phase = packing.detail_phase("pack trees");
        data.trees
            .iter()
            .enumerate()
            .map(|(ti, t)| {
                let packed = pack_tree(t, options);
                packing.set_fraction((ti + 1) as f32 / data.trees.len() as f32);
                packed
            })
            .collect::<Result<_, _>>()?
    };
    packing.check_cancel()?;
    sfmtool_archive_io::write_atomically(path, |file| {
        write_into(
            file,
            data,
            sources,
            options,
            &packed,
            Shares {
                heading: &heading,
                descriptors: &descriptors,
                geometry: &geometry,
                chunks: &chunks,
            },
        )
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
    // Every scalar of the corpus, every geometry row, and every tree's feature
    // permutation is read here, which is a whole-corpus pass whichever way it is
    // written. The parallel forms below produce the same verdict as the serial
    // ones: each is a pure predicate over independent rows, and where a tree or
    // an origin is at fault the results are collected in input order so the
    // error names the same one every run.
    if data.vectors.par_iter().any(|&v| !v.is_finite()) {
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
    for outcome in data
        .trees
        .par_iter()
        .enumerate()
        .map(|(ti, tree)| validate_tree(tree, data.feature_count, data.dimension, ti))
        .collect::<Vec<_>>()
    {
        outcome?;
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
            .par_iter()
            .any(|row| row.iter().flatten().any(|v| !v.is_finite()))
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
        if !src
            .image_names
            .iter()
            .all(|n| valid_relative_posix(n) && names.insert(n))
        {
            return Err(KdfError::InvalidFormat(
                "image names must be unique relative POSIX paths".into(),
            ));
        }
        if src
            .origins
            .par_iter()
            .any(|o| o.image_index as usize >= src.image_names.len())
        {
            return Err(KdfError::InvalidFormat(
                "origin image index is out of range".into(),
            ));
        }
        // A pair is unique iff no two are equal, which a sort answers without a
        // hash table: the set this replaces held one entry per corpus feature,
        // and at a few million features its probing cost more than ordering the
        // pairs does. The packing is order-preserving on the pair, so equal
        // packed values mean equal pairs and nothing else.
        let mut pairs: Vec<u64> = src
            .origins
            .par_iter()
            .map(|o| (o.image_index as u64) << 32 | o.image_feature_index as u64)
            .collect();
        pairs.par_sort_unstable();
        if pairs.par_windows(2).any(|w| w[0] == w[1]) {
            return Err(KdfError::InvalidFormat(
                "duplicate image-feature origin pair".into(),
            ));
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
    ids.par_sort_unstable();
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

/// The range each of [`write_into`]'s four stages reports within, carved out of
/// the caller's by [`STAGE_SHARES`].
///
/// One struct rather than four parameters, so the stage a range belongs to is
/// named where it is used rather than counted off at the call.
struct Shares<'p, 'a> {
    heading: &'p Progress<'a>,
    descriptors: &'p Progress<'a>,
    geometry: &'p Progress<'a>,
    chunks: &'p Progress<'a>,
}

fn write_into<W: Write + Seek, S: KdfScalar>(
    writer: W,
    data: &KdfForestData<'_, S>,
    sources: Option<&KdfSiftSources>,
    options: &KdfWriteOptions,
    packed: &[Vec<PackedChunk<S>>],
    shares: Shares<'_, '_>,
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
    let mut section_digests = SectionDigests::new();
    section_digests.push(metadata_digest);
    let heading_phase = shares.heading.detail_phase("heading");

    let (images_digest, origins_digest) = if let Some(src) = sources {
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
        // The image table is written; the origins are the rest of the heading,
        // bar the row map the next stage opens with.
        shares.heading.set_fraction(0.2);
        let mut ods = SectionDigests::new();
        let origin_blocks = src.origins.len().div_ceil(options.origin_block_rows);
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
            // A block is 131 072 rows, so a capture has a couple of dozen of
            // them and each one is worth a report of its own.
            shares
                .heading
                .set_fraction(0.2 + 0.6 * (b + 1) as f32 / origin_blocks as f32);
        }
        shares.heading.check_cancel()?;
        let origins_digest = ods.finish();
        section_digests.push(images_digest);
        section_digests.push(origins_digest);
        (Some(images_digest), Some(origins_digest))
    } else {
        (None, None)
    };

    let (storage_digest, descriptors_digest, geometry_digest) = {
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
        shares.heading.set_fraction(1.0);
        shares.heading.check_cancel()?;
        drop(heading_phase);
        // One container entry of independent per-block frames, plus the offsets
        // that address them. What goes away against a standalone entry per block
        // is one ZIP directory record apiece, which at small block sizes is most
        // of the file's entries and most of its open cost.
        //
        // Stream frames directly: buffering the container duplicates the entire
        // compressed corpus in memory. ZIP64 permits a stream larger than 4 GiB.
        zip.start_file(
            corpus_entry_name::<S>(data.feature_count, data.dimension),
            zip::write::SimpleFileOptions::default()
                .compression_method(zip::CompressionMethod::Stored)
                .large_file(true),
        )?;
        let descriptor_phase = shares.descriptors.detail_phase("descriptor blocks");
        let (descriptors_digest, descriptor_offsets) = write_blocks(
            &mut zip,
            order,
            q,
            options.compression_level,
            shares.descriptors,
            |ids, raw| {
                for &id in ids {
                    let base = id as usize * data.dimension;
                    raw.extend_from_slice(bytemuck::cast_slice(
                        &data.vectors[base..base + data.dimension],
                    ));
                }
            },
        )?;
        section_digests.push(descriptors_digest);
        drop(descriptor_phase);
        let offsets_raw: &[u8] = bytemuck::cast_slice(descriptor_offsets.as_slice());
        write_binary_entry(
            &mut zip,
            &block_offsets_entry_name(descriptor_offsets.len()),
            offsets_raw,
            options.compression_level,
        )?;
        let geometry_digest = if let Some(src) = sources {
            zip.start_file(
                geometry_entry_name(data.feature_count),
                zip::write::SimpleFileOptions::default()
                    .compression_method(zip::CompressionMethod::Stored)
                    .large_file(true),
            )?;
            let geometry_phase = shares.geometry.detail_phase("geometry blocks");
            let (digest, offsets) = write_blocks(
                &mut zip,
                order,
                q,
                options.compression_level,
                shares.geometry,
                |ids, raw| {
                    for &id in ids {
                        raw.extend_from_slice(bytemuck::bytes_of(&src.geometry[id as usize]));
                    }
                },
            )?;
            section_digests.push(digest);
            drop(geometry_phase);
            let offsets_raw: &[u8] = bytemuck::cast_slice(offsets.as_slice());
            write_binary_entry(
                &mut zip,
                &geometry_block_offsets_entry_name(offsets.len()),
                offsets_raw,
                options.compression_level,
            )?;
            Some(digest)
        } else {
            None
        };
        (sd, descriptors_digest, geometry_digest)
    };

    let mut trees_digest = SectionDigests::new();
    let total_chunks: usize = packed.iter().map(Vec::len).sum();
    let mut written_chunks = 0usize;
    let chunk_phase = shares.chunks.detail_phase("tree chunks");
    for (ti, chunks) in packed.iter().enumerate() {
        // A tree is the batch here: there are hundreds of chunks rather than
        // hundreds of thousands, and a chunk is about a megabyte, so a whole
        // tree's frames are tens of megabytes held for as long as it takes to
        // write them out in order.
        let encoded = chunks
            .par_iter()
            .map_init(
                || None::<zstd::bulk::Compressor<'static>>,
                |slot, chunk| -> Result<(Vec<u8>, u128), KdfError> {
                    if slot.is_none() {
                        *slot = Some(zstd::bulk::Compressor::new(options.compression_level)?);
                    }
                    let payload = chunk_payload(chunk);
                    let digest = xxh3_128(&payload);
                    let frame = slot.as_mut().expect("initialized").compress(&payload)?;
                    Ok((frame, digest))
                },
            )
            .collect::<Result<Vec<_>, _>>()?;
        for (ci, ((frame, digest), chunk)) in encoded.iter().zip(chunks).enumerate() {
            zip.start_file(
                chunk_entry_name::<S>(ti, ci, chunk.nodes.len(), chunk.feature_ids.len()),
                zip::write::SimpleFileOptions::default()
                    .compression_method(zip::CompressionMethod::Stored),
            )?;
            zip.write_all(frame)?;
            trees_digest.push(*digest);
            written_chunks += 1;
            shares
                .chunks
                .set_fraction(written_chunks as f32 / total_chunks as f32);
        }
        shares.chunks.check_cancel()?;
    }
    drop(chunk_phase);
    let trees_digest = trees_digest.finish();
    section_digests.push(trees_digest);

    let hashes = ContentHash {
        metadata_xxh128: format_hash(metadata_digest),
        images_xxh128: images_digest.map(format_hash),
        origins_xxh128: origins_digest.map(format_hash),
        storage_rows_xxh128: format_hash(storage_digest),
        descriptors_xxh128: format_hash(descriptors_digest),
        geometry_xxh128: geometry_digest.map(format_hash),
        trees_xxh128: format_hash(trees_digest),
        content_xxh128: format_hash(section_digests.finish()),
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

#[cfg(test)]
mod tests;

/// One chunk's wire payload: node columns, then splits, then feature IDs.
///
/// The three integer arrays share one entry because they are always read
/// together — decoding a node needs the columns and the splits, and reaching a
/// leaf needs the IDs — and they compress alike. Vectors stay separate despite
/// also always being read with them, because bulk descriptor bytes in the same
/// zstd frame as these columns make both compress worse.
fn chunk_payload<S: KdfScalar>(chunk: &PackedChunk<S>) -> Vec<u8> {
    let m = chunk.nodes.len();
    let mut columns = vec![0u32; NODE_COLUMNS * m];
    let mut splits = vec![S::ZERO; m];
    for (i, (&logical, node)) in chunk.logical_nodes.iter().zip(&chunk.nodes).enumerate() {
        columns[i] = match node {
            DecodedNode::Internal { .. } => 0,
            DecodedNode::Leaf { .. } => 1,
        };
        columns[m + i] = logical;
        match *node {
            DecodedNode::Internal {
                split_dimension,
                split,
                left,
                right,
            } => {
                columns[2 * m + i] = split_dimension as u32;
                columns[3 * m + i] = left.chunk;
                columns[4 * m + i] = left.local;
                columns[5 * m + i] = left.logical;
                columns[6 * m + i] = right.chunk;
                columns[7 * m + i] = right.local;
                columns[8 * m + i] = right.logical;
                splits[i] = split;
            }
            DecodedNode::Leaf { start, .. } => columns[9 * m + i] = start,
        }
    }
    let nb: &[u8] = bytemuck::cast_slice(columns.as_slice());
    let sb: &[u8] = bytemuck::cast_slice(splits.as_slice());
    let fb: &[u8] = bytemuck::cast_slice(chunk.feature_ids.as_slice());
    let mut payload = Vec::with_capacity(nb.len() + sb.len() + fb.len());
    payload.extend_from_slice(nb);
    payload.extend_from_slice(sb);
    payload.extend_from_slice(fb);
    payload
}

/// Stream one blocked section into the open container entry.
///
/// Returns the section's digest and its `blocks + 1` frame boundaries. The
/// blocks are compressed and hashed a batch at a time across the thread pool and
/// then written in block order, so the entry's bytes, the boundaries and the
/// digest are what a single thread would have produced. Cancellation and the
/// stage fraction are read on this thread between batches, once every batch
/// rather than once every couple of KiB of compression.
fn write_blocks<W: Write + Seek, F>(
    zip: &mut ZipWriter<W>,
    order: &[u32],
    rows_per_block: usize,
    level: i32,
    progress: &Progress<'_>,
    gather: F,
) -> Result<(u128, Vec<u64>), KdfError>
where
    F: Fn(&[u32], &mut Vec<u8>) + Sync,
{
    let blocks = order.len().div_ceil(rows_per_block);
    let batch = block_batch(blocks);
    let mut digests = SectionDigests::new();
    let mut offsets = Vec::with_capacity(blocks + 1);
    offsets.push(0u64);
    let mut stored = 0u64;
    // The corpus order is sliced a batch at a time rather than all at once: a
    // slice per block would be sixteen bytes a block held for the whole stage,
    // which is the one part of this loop that would grow with the corpus.
    for (bi, group) in order.chunks(rows_per_block * batch).enumerate() {
        let ids: Vec<&[u32]> = group.chunks(rows_per_block).collect();
        for (frame, digest) in encode_batch(&ids, level, &gather)? {
            zip.write_all(&frame)?;
            stored += frame.len() as u64;
            offsets.push(stored);
            digests.push(digest);
        }
        let done = ((bi + 1) * batch).min(blocks);
        progress.set_fraction(done as f32 / blocks as f32);
        progress.check_cancel()?;
    }
    progress.set_fraction(1.0);
    Ok((digests.finish(), offsets))
}
