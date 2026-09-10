// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashMap, HashSet};
use std::io::Read;
use std::path::Path;
use std::sync::{Arc, Mutex, OnceLock};

use xxhash_rust::xxh3::{xxh3_128, Xxh3};
use zip::ZipArchive;

use crate::cache::{Cache, CacheKey, Cached};
use crate::types::*;

const NODE_COLUMNS: usize = 10;

/// An open immutable `.kdf` snapshot with lazy, integrity-checked payload access.
pub struct KdfFile<S: KdfScalar> {
    archive: Mutex<ZipArchive<std::fs::File>>,
    entries: HashMap<String, u64>,
    metadata: Metadata,
    hashes: ContentHash,
    storage_rows: Option<Vec<u32>>,
    image_table: OnceLock<KdfImageTable>,
    cache: Arc<Cache<S>>,
    max_metadata_bytes: usize,
    max_compressed_bytes: usize,
    max_leaf_features: usize,
    image_count: Option<usize>,
}

impl<S: KdfScalar> KdfFile<S> {
    /// Open a local seekable archive without decoding tree or descriptor chunks.
    pub fn open(path: &Path, options: LazyKdForestOptions) -> Result<Self, KdfError> {
        if options.query_workers == 0 {
            return Err(KdfError::ResourceLimit(
                "query_workers must be positive".into(),
            ));
        }
        if options.cache_bytes == 0
            || options.max_in_flight_bytes == 0
            || options.max_compressed_bytes == 0
            || options.max_metadata_bytes == 0
            || options.max_chunk_bytes == 0
        {
            return Err(KdfError::ResourceLimit(
                "all byte limits must be positive".into(),
            ));
        }
        let file = std::fs::File::open(path)?;
        let mut archive = ZipArchive::new(file)?;
        let mut entries = HashMap::new();
        let mut directory_bytes = 0usize;
        for i in 0..archive.len() {
            let entry = archive.by_index(i)?;
            if entry.is_dir() {
                return Err(KdfError::InvalidFormat(format!(
                    "directory ZIP entry is forbidden: {}",
                    entry.name()
                )));
            }
            directory_bytes = directory_bytes
                .checked_add(entry.name().len() + 64)
                .ok_or_else(|| {
                    KdfError::ResourceLimit("ZIP directory accounting overflow".into())
                })?;
            if directory_bytes > options.max_metadata_bytes {
                return Err(KdfError::ResourceLimit(
                    "ZIP directory exceeds max_metadata_bytes".into(),
                ));
            }
            if entry.compression() != zip::CompressionMethod::Stored {
                return Err(KdfError::InvalidFormat(format!(
                    "ZIP entry is not STORE: {}",
                    entry.name()
                )));
            }
            if entries
                .insert(entry.name().to_owned(), entry.compressed_size())
                .is_some()
            {
                return Err(KdfError::InvalidFormat(format!(
                    "duplicate ZIP entry: {}",
                    entry.name()
                )));
            }
        }
        let remaining = options
            .max_metadata_bytes
            .checked_sub(directory_bytes)
            .ok_or_else(|| {
                KdfError::ResourceLimit("ZIP directory exhausts metadata budget".into())
            })?;
        let metadata_raw = read_bounded_json_raw(
            &mut archive,
            &entries,
            "metadata.json.zst",
            remaining.min(options.max_compressed_bytes),
        )?;
        let metadata: Metadata = serde_json::from_slice(&metadata_raw)?;
        let remaining = remaining
            .checked_sub(metadata_raw.len())
            .ok_or_else(|| KdfError::ResourceLimit("metadata exceeds budget".into()))?;
        let hash_raw = read_bounded_json_raw(
            &mut archive,
            &entries,
            "content_hash.json.zst",
            remaining.min(options.max_compressed_bytes),
        )?;
        let hashes: ContentHash = serde_json::from_slice(&hash_raw)?;
        validate_metadata::<S>(&metadata, &hashes, &options)?;
        if hash_string(xxh3_128(&metadata_raw)) != hashes.metadata_xxh128 {
            return Err(KdfError::Integrity("metadata hash mismatch".into()));
        }
        let image_count = if metadata.feature_source == "sift_files" {
            let remaining = remaining.checked_sub(hash_raw.len()).ok_or_else(|| {
                KdfError::ResourceLimit("metadata and hash directory exceed budget".into())
            })?;
            let raw = read_bounded_json_raw(
                &mut archive,
                &entries,
                "images/metadata.json.zst",
                remaining.min(options.max_compressed_bytes),
            )?;
            Some(serde_json::from_slice::<ImagesMetadata>(&raw)?.image_count as usize)
        } else {
            None
        };
        let expected = expected_entries::<S>(&metadata, image_count)?;
        let actual: HashSet<String> = entries.keys().cloned().collect();
        if expected != actual {
            return Err(KdfError::InvalidFormat(describe_entry_difference(
                &expected, &actual,
            )));
        }

        let storage_rows = if metadata.descriptor_storage == "shared" {
            let bytes = (metadata.feature_count as usize)
                .checked_mul(4)
                .ok_or_else(|| KdfError::ResourceLimit("address map size overflow".into()))?;
            let validation_scratch = metadata.feature_count as usize;
            if bytes
                .checked_add(validation_scratch)
                .is_none_or(|v| v > options.max_address_map_bytes)
            {
                return Err(KdfError::ResourceLimit(format!(
                    "shared row map and validation scratch require more than {} bytes",
                    options.max_address_map_bytes
                )));
            }
            let name = format!(
                "features/storage_rows.{}.uint32.zst",
                metadata.feature_count
            );
            let raw = read_exact_raw(
                &mut archive,
                &entries,
                &name,
                bytes,
                options.max_compressed_bytes,
            )?;
            let expected_hash = hashes
                .storage_rows_xxh128
                .as_deref()
                .ok_or_else(|| KdfError::InvalidFormat("missing storage_rows_xxh128".into()))?;
            if hash_string(xxh3_128(&raw)) != expected_hash {
                return Err(KdfError::Integrity(
                    "shared storage-row hash mismatch".into(),
                ));
            }
            let rows: Vec<u32> = bytes_to_pod(&name, &raw, metadata.feature_count as usize)?;
            validate_permutation(&rows, metadata.feature_count as usize, "storage row map")?;
            Some(rows)
        } else {
            None
        };
        let address_map_bytes = storage_rows
            .as_ref()
            .map_or(0, |v| std::mem::size_of_val(v.as_slice()));
        let cache = Cache::new(
            options.cache_bytes,
            options.max_in_flight_bytes,
            address_map_bytes,
        );
        Ok(Self {
            archive: Mutex::new(archive),
            entries,
            metadata,
            hashes,
            storage_rows,
            image_table: OnceLock::new(),
            cache,
            max_metadata_bytes: options.max_metadata_bytes,
            max_compressed_bytes: options.max_compressed_bytes,
            max_leaf_features: options.max_leaf_features,
            image_count,
        })
    }

    pub fn len(&self) -> usize {
        self.metadata.feature_count as usize
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn dim(&self) -> usize {
        self.metadata.dimension as usize
    }
    pub fn tree_count(&self) -> usize {
        self.metadata.trees.len()
    }
    pub fn scalar_type(&self) -> &str {
        &self.metadata.scalar_type
    }
    pub fn is_shared(&self) -> bool {
        self.storage_rows.is_some()
    }
    pub fn options_max_leaf_features(&self) -> usize {
        self.max_leaf_features
    }
    pub fn root(&self, tree: usize) -> Option<NodeAddress> {
        self.metadata.trees.get(tree)?.root.map(|v| NodeAddress {
            chunk: v[0],
            local: v[1],
            logical: 0,
        })
    }
    pub fn io_stats(&self) -> KdfIoStats {
        self.cache.stats()
    }
    /// Zero the cumulative I/O counters, keeping resident and in-flight bytes.
    ///
    /// The peaks restart from the current gauges rather than from zero. A
    /// benchmark uses this to separate an open from the queries that follow it,
    /// or a cold pass from a warm one, without reopening the file.
    pub fn reset_io_stats(&self) {
        self.cache.reset_counters();
    }

    /// Read a single node, retaining no cache pin on return.
    pub fn node(&self, tree: u32, address: NodeAddress) -> Result<DecodedNode<S>, KdfError> {
        let pin = self.tree_chunk(tree, address.chunk)?;
        let Cached::Tree(chunk) = &*pin else {
            unreachable!()
        };
        let node = *chunk
            .nodes
            .get(address.local as usize)
            .ok_or_else(|| KdfError::InvalidFormat("node local index out of range".into()))?;
        // The parent-stored logical ID must agree with the addressed child.
        let actual = *chunk
            .logical_node_ids
            .get(address.local as usize)
            .ok_or_else(|| KdfError::InvalidFormat("node local index out of range".into()))?;
        if actual != address.logical {
            return Err(KdfError::InvalidFormat(
                "child logical ID does not match addressed node".into(),
            ));
        }
        Ok(node)
    }

    /// Copy a leaf's ordered members and local vectors, releasing its chunk pin.
    pub fn leaf(&self, tree: u32, address: NodeAddress) -> Result<DecodedLeaf<S>, KdfError> {
        let pin = self.tree_chunk(tree, address.chunk)?;
        let Cached::Tree(chunk) = &*pin else {
            unreachable!()
        };
        let node = chunk
            .nodes
            .get(address.local as usize)
            .ok_or_else(|| KdfError::InvalidFormat("leaf local index out of range".into()))?;
        let DecodedNode::Leaf { start, len } = *node else {
            return Err(KdfError::InvalidFormat("address is not a leaf".into()));
        };
        if len as usize > self.max_leaf_features {
            return Err(KdfError::ResourceLimit(format!(
                "leaf has {len} features; max_leaf_features={}",
                self.max_leaf_features
            )));
        }
        let range = start as usize..(start + len) as usize;
        let ids = chunk.feature_ids[range.clone()].to_vec();
        let vectors = chunk.vectors.as_ref().map(|v| {
            let d = self.dim();
            v[range.start * d..range.end * d].to_vec()
        });
        Ok(DecodedLeaf {
            feature_ids: ids,
            vectors,
        })
    }

    /// Copy one vector from the shared corpus. Tree-local callers use `leaf`.
    pub fn shared_vector(&self, feature_id: u32) -> Result<Vec<S>, KdfError> {
        let rows = self
            .storage_rows
            .as_ref()
            .ok_or_else(|| KdfError::InvalidFormat("forest is tree-local".into()))?;
        let row = *rows
            .get(feature_id as usize)
            .ok_or_else(|| KdfError::InvalidQuery("feature ID out of range".into()))?
            as usize;
        let q = self.metadata.descriptor_block_rows.expect("validated") as usize;
        let block = row / q;
        let within = row % q;
        let pin = self.descriptor_block(block as u32)?;
        let Cached::Descriptor(vectors) = &*pin else {
            unreachable!()
        };
        let base = within * self.dim();
        Ok(vectors[base..base + self.dim()].to_vec())
    }

    /// Resolve mappings in request order, including repeated feature IDs.
    pub fn resolve_origins(
        &self,
        feature_ids: &[u32],
    ) -> Result<Option<Vec<FeatureOrigin>>, KdfError> {
        if self.metadata.feature_source == "none" {
            return Ok(None);
        }
        let rows = self.metadata.origin_block_rows.expect("validated") as usize;
        let mut out = Vec::with_capacity(feature_ids.len());
        for &id in feature_ids {
            if id as usize >= self.len() {
                return Err(KdfError::InvalidQuery(format!(
                    "feature ID {id} is out of range"
                )));
            }
            let block = id as usize / rows;
            let within = id as usize % rows;
            let pin = self.origin_block(block as u32)?;
            let Cached::Origin(origins) = &*pin else {
                unreachable!()
            };
            out.push(origins[within]);
        }
        Ok(Some(out))
    }

    /// Load the image table on first request. Normal ANN never calls this.
    pub fn image_table(&self) -> Result<Option<&KdfImageTable>, KdfError> {
        if self.metadata.feature_source == "none" {
            return Ok(None);
        }
        if let Some(v) = self.image_table.get() {
            return Ok(Some(v));
        }
        let count = self.image_count.expect("validated");
        let names_name = "images/names.json.zst";
        let mut archive = self.archive.lock().unwrap();
        let names_raw = read_bounded_json_raw(
            &mut archive,
            &self.entries,
            names_name,
            self.max_metadata_bytes,
        )?;
        let names: Vec<String> = serde_json::from_slice(&names_raw)?;
        if names.len() != count {
            return Err(KdfError::ShapeMismatch("image names count mismatch".into()));
        }
        if names_raw
            .len()
            .checked_add(count.saturating_mul(32))
            .is_none_or(|v| v > self.max_metadata_bytes)
        {
            return Err(KdfError::ResourceLimit(
                "decoded image table exceeds max_metadata_bytes".into(),
            ));
        }
        let f_name = format!("images/feature_tool_hashes.{count}.uint128.zst");
        let s_name = format!("images/sift_content_hashes.{count}.uint128.zst");
        let f_raw = read_exact_raw(
            &mut archive,
            &self.entries,
            &f_name,
            count * 16,
            self.max_compressed_bytes,
        )?;
        let s_raw = read_exact_raw(
            &mut archive,
            &self.entries,
            &s_name,
            count * 16,
            self.max_compressed_bytes,
        )?;
        let meta_raw = read_bounded_json_raw(
            &mut archive,
            &self.entries,
            "images/metadata.json.zst",
            self.max_metadata_bytes,
        )?;
        let mut h = Xxh3::new();
        h.update(&f_raw);
        h.update(&meta_raw);
        h.update(&names_raw);
        h.update(&s_raw);
        if hash_string(h.digest128()) != self.hashes.images_xxh128.as_deref().expect("validated") {
            return Err(KdfError::Integrity("images section hash mismatch".into()));
        }
        let table = KdfImageTable {
            names,
            feature_tool_hashes: bytes_to_hashes(&f_name, &f_raw, count)?,
            sift_content_hashes: bytes_to_hashes(&s_name, &s_raw, count)?,
        };
        drop(archive);
        let _ = self.image_table.set(table);
        Ok(self.image_table.get())
    }

    pub(crate) fn metadata(&self) -> &Metadata {
        &self.metadata
    }

    fn tree_chunk(&self, tree: u32, chunk: u32) -> Result<crate::cache::CachePin<S>, KdfError> {
        let meta = self
            .metadata
            .trees
            .get(tree as usize)
            .and_then(|t| t.chunks.get(chunk as usize))
            .ok_or_else(|| KdfError::InvalidFormat("tree chunk address out of range".into()))?;
        let declared = usize::try_from(meta.decoded_bytes)
            .map_err(|_| KdfError::ResourceLimit("chunk decoded size exceeds usize".into()))?;
        self.cache
            .get_or_load(CacheKey::Tree(tree, chunk), declared, || {
                self.load_tree_chunk(tree, chunk, meta)
            })
    }

    fn load_tree_chunk(
        &self,
        tree: u32,
        chunk: u32,
        meta: &ChunkMetadata,
    ) -> Result<(Cached<S>, u64), KdfError> {
        let m = meta.node_count as usize;
        let p = meta.feature_count as usize;
        let prefix = format!("trees/{tree}/chunks/{chunk}");
        let names = [
            format!("{prefix}/nodes.{NODE_COLUMNS}.{m}.uint32.zst"),
            format!("{prefix}/splits.{m}.{}.zst", S::TYPE_NAME),
            format!("{prefix}/feature_ids.{p}.uint32.zst"),
        ];
        let mut archive = self.archive.lock().unwrap();
        let (nodes_raw, a) = read_exact_counted(
            &mut archive,
            &self.entries,
            &names[0],
            NODE_COLUMNS * m * 4,
            self.max_compressed_bytes,
        )?;
        let (splits_raw, b) = read_exact_counted(
            &mut archive,
            &self.entries,
            &names[1],
            m * std::mem::size_of::<S>(),
            self.max_compressed_bytes,
        )?;
        let (ids_raw, c) = read_exact_counted(
            &mut archive,
            &self.entries,
            &names[2],
            p * 4,
            self.max_compressed_bytes,
        )?;
        let (vectors_raw, d) = if self.metadata.descriptor_storage == "tree_local" {
            let n = format!("{prefix}/vectors.{p}.{}.{}.zst", self.dim(), S::TYPE_NAME);
            let (v, z) = read_exact_counted(
                &mut archive,
                &self.entries,
                &n,
                p * self.dim() * std::mem::size_of::<S>(),
                self.max_compressed_bytes,
            )?;
            (Some((n, v)), z)
        } else {
            (None, 0)
        };
        drop(archive);
        let mut h = Xxh3::new();
        h.update(&nodes_raw);
        h.update(&splits_raw);
        h.update(&ids_raw);
        if let Some((_, raw)) = &vectors_raw {
            h.update(raw);
        }
        let expected = &self.hashes.chunks_xxh128[tree as usize][chunk as usize];
        if hash_string(h.digest128()) != *expected {
            return Err(KdfError::Integrity(format!(
                "tree {tree} chunk {chunk} hash mismatch"
            )));
        }
        let columns: Vec<u32> = bytes_to_pod(&names[0], &nodes_raw, NODE_COLUMNS * m)?;
        let splits: Vec<S> = bytes_to_pod(&names[1], &splits_raw, m)?;
        let feature_ids: Vec<u32> = bytes_to_pod(&names[2], &ids_raw, p)?;
        let vectors: Option<Vec<S>> = vectors_raw
            .map(|(name, raw)| bytes_to_pod(&name, &raw, p * self.dim()))
            .transpose()?;
        let nodes = decode_nodes(&self.metadata, tree as usize, &columns, &splits, m, p)?;
        if feature_ids
            .iter()
            .any(|&id| id >= self.metadata.feature_count)
        {
            return Err(KdfError::InvalidFormat("feature ID out of range".into()));
        }
        if vectors
            .as_ref()
            .is_some_and(|v| v.iter().any(|&x| !x.is_finite()))
        {
            return Err(KdfError::InvalidFormat(
                "tree chunk contains non-finite vector".into(),
            ));
        }
        let logical_node_ids = (0..m).map(|i| columns[m + i]).collect();
        Ok((
            Cached::Tree(DecodedTreeChunk {
                logical_node_ids,
                nodes,
                feature_ids,
                vectors,
                decoded_bytes: meta.decoded_bytes as usize,
            }),
            a + b + c + d,
        ))
    }

    fn descriptor_block(&self, block: u32) -> Result<crate::cache::CachePin<S>, KdfError> {
        let q = self.metadata.descriptor_block_rows.expect("validated") as usize;
        let start = block as usize * q;
        if start >= self.len() {
            return Err(KdfError::InvalidFormat(
                "descriptor block out of range".into(),
            ));
        }
        let r = q.min(self.len() - start);
        let declared = r * self.dim() * std::mem::size_of::<S>();
        self.cache
            .get_or_load(CacheKey::Descriptor(block), declared, || {
                let name = format!(
                    "features/blocks/{block}/vectors.{r}.{}.{}.zst",
                    self.dim(),
                    S::TYPE_NAME
                );
                let mut archive = self.archive.lock().unwrap();
                let (raw, compressed) = read_exact_counted(
                    &mut archive,
                    &self.entries,
                    &name,
                    declared,
                    self.max_compressed_bytes,
                )?;
                drop(archive);
                if hash_string(xxh3_128(&raw))
                    != self
                        .hashes
                        .descriptor_blocks_xxh128
                        .as_ref()
                        .expect("validated")[block as usize]
                {
                    return Err(KdfError::Integrity(format!(
                        "descriptor block {block} hash mismatch"
                    )));
                }
                let values: Vec<S> = bytes_to_pod(&name, &raw, r * self.dim())?;
                if values.iter().any(|&x| !x.is_finite()) {
                    return Err(KdfError::InvalidFormat(
                        "descriptor block contains non-finite vector".into(),
                    ));
                }
                Ok((Cached::Descriptor(values), compressed))
            })
    }

    fn origin_block(&self, block: u32) -> Result<crate::cache::CachePin<S>, KdfError> {
        let q = self.metadata.origin_block_rows.expect("validated") as usize;
        let start = block as usize * q;
        if start >= self.len() {
            return Err(KdfError::InvalidQuery("origin block out of range".into()));
        }
        let r = q.min(self.len() - start);
        let declared = r * 8;
        self.cache
            .get_or_load(CacheKey::Origin(block), declared, || {
                let a_name = format!("origins/{block}/image_indexes.{r}.uint32.zst");
                let f_name = format!("origins/{block}/image_feature_indexes.{r}.uint32.zst");
                let mut archive = self.archive.lock().unwrap();
                let (a_raw, a) = read_exact_counted(
                    &mut archive,
                    &self.entries,
                    &a_name,
                    r * 4,
                    self.max_compressed_bytes,
                )?;
                let (f_raw, f) = read_exact_counted(
                    &mut archive,
                    &self.entries,
                    &f_name,
                    r * 4,
                    self.max_compressed_bytes,
                )?;
                drop(archive);
                let mut h = Xxh3::new();
                h.update(&a_raw);
                h.update(&f_raw);
                if hash_string(h.digest128())
                    != self.hashes.origins_xxh128.as_ref().expect("validated")[block as usize]
                {
                    return Err(KdfError::Integrity(format!(
                        "origin block {block} hash mismatch"
                    )));
                }
                let images: Vec<u32> = bytes_to_pod(&a_name, &a_raw, r)?;
                let features: Vec<u32> = bytes_to_pod(&f_name, &f_raw, r)?;
                let origins: Vec<FeatureOrigin> = images
                    .into_iter()
                    .zip(features)
                    .map(|(image_index, image_feature_index)| FeatureOrigin {
                        image_index,
                        image_feature_index,
                    })
                    .collect();
                if origins
                    .iter()
                    .any(|o| o.image_index as usize >= self.image_count.expect("validated"))
                {
                    return Err(KdfError::InvalidFormat(
                        "origin image index out of range".into(),
                    ));
                }
                Ok((Cached::Origin(origins), a + f))
            })
    }
}

fn validate_metadata<S: KdfScalar>(
    m: &Metadata,
    h: &ContentHash,
    o: &LazyKdForestOptions,
) -> Result<(), KdfError> {
    if m.format != "kdf" || m.version != KDF_FORMAT_VERSION || m.metric != "squared_l2" {
        return Err(KdfError::InvalidFormat(
            "unsupported format, version, or metric".into(),
        ));
    }
    if m.scalar_type != S::TYPE_NAME {
        return Err(KdfError::ScalarType {
            file: m.scalar_type.clone(),
            requested: S::TYPE_NAME,
        });
    }
    if m.dimension == 0 || m.trees.is_empty() || m.node_kinds != ["internal", "leaf"] {
        return Err(KdfError::InvalidFormat(
            "invalid dimension, tree count, or node legend".into(),
        ));
    }
    if !matches!(m.descriptor_storage.as_str(), "tree_local" | "shared")
        || !matches!(m.feature_source.as_str(), "none" | "sift_files")
    {
        return Err(KdfError::InvalidFormat(
            "unsupported storage or feature source".into(),
        ));
    }
    if (m.descriptor_storage == "shared") != m.descriptor_block_rows.is_some()
        || m.descriptor_block_rows == Some(0)
    {
        return Err(KdfError::InvalidFormat(
            "invalid descriptor_block_rows presence/value".into(),
        ));
    }
    if (m.feature_source == "sift_files")
        != (m.origin_block_rows.is_some() && m.workspace.is_some())
        || m.origin_block_rows == Some(0)
    {
        return Err(KdfError::InvalidFormat(
            "invalid SIFT source metadata".into(),
        ));
    }
    if h.chunks_xxh128.len() != m.trees.len() {
        return Err(KdfError::InvalidFormat(
            "chunk hash tree count mismatch".into(),
        ));
    }
    let mut largest = 0usize;
    for (ti, tree) in m.trees.iter().enumerate() {
        if h.chunks_xxh128[ti].len() != tree.chunks.len() {
            return Err(KdfError::InvalidFormat("chunk hash count mismatch".into()));
        }
        if m.feature_count == 0 {
            if tree.root.is_some() || !tree.chunks.is_empty() {
                return Err(KdfError::InvalidFormat(
                    "empty forest contains tree data".into(),
                ));
            }
        } else if tree.root != Some([0, 0]) || tree.chunks.is_empty() {
            return Err(KdfError::InvalidFormat(
                "nonempty tree root must be chunk 0 node 0".into(),
            ));
        }
        for c in &tree.chunks {
            if c.node_count == 0 {
                return Err(KdfError::InvalidFormat("empty chunk".into()));
            }
            let decoded = usize::try_from(c.decoded_bytes)
                .map_err(|_| KdfError::ResourceLimit("chunk size exceeds usize".into()))?;
            if decoded > o.max_chunk_bytes {
                return Err(KdfError::ResourceLimit(format!(
                    "declared chunk {decoded} exceeds max_chunk_bytes"
                )));
            }
            largest = largest.max(decoded);
        }
    }
    let row_bytes = m.dimension as usize * std::mem::size_of::<S>();
    if let Some(q) = m.descriptor_block_rows {
        largest = largest.max((q as usize).min(m.feature_count as usize) * row_bytes);
    }
    if let Some(q) = m.origin_block_rows {
        largest = largest.max((q as usize).min(m.feature_count as usize) * 8);
    }
    if largest > o.max_chunk_bytes {
        return Err(KdfError::ResourceLimit(format!(
            "declared cache item {largest} exceeds max_chunk_bytes"
        )));
    }
    if largest > o.cache_bytes || largest > o.max_in_flight_bytes {
        return Err(KdfError::ResourceLimit(format!(
            "largest declared item {largest} does not fit cache/in-flight limits"
        )));
    }
    validate_hash_shape(h, m)?;
    Ok(())
}

fn validate_hash_shape(h: &ContentHash, m: &Metadata) -> Result<(), KdfError> {
    for s in std::iter::once(&h.metadata_xxh128)
        .chain(std::iter::once(&h.content_xxh128))
        .chain(h.chunks_xxh128.iter().flatten())
    {
        parse_hash(s)?;
    }
    let shared = m.descriptor_storage == "shared";
    if shared != (h.storage_rows_xxh128.is_some() && h.descriptor_blocks_xxh128.is_some()) {
        return Err(KdfError::InvalidFormat(
            "shared hash fields mismatch layout".into(),
        ));
    }
    let descriptor_blocks = m
        .descriptor_block_rows
        .map_or(0, |q| (m.feature_count as usize).div_ceil(q as usize));
    if h.descriptor_blocks_xxh128.as_ref().map_or(0, Vec::len) != descriptor_blocks {
        return Err(KdfError::InvalidFormat(
            "descriptor block hash count mismatch".into(),
        ));
    }
    let sift = m.feature_source == "sift_files";
    if sift != (h.images_xxh128.is_some() && h.origins_xxh128.is_some()) {
        return Err(KdfError::InvalidFormat(
            "source hash fields mismatch feature source".into(),
        ));
    }
    let origin_blocks = m
        .origin_block_rows
        .map_or(0, |q| (m.feature_count as usize).div_ceil(q as usize));
    if h.origins_xxh128.as_ref().map_or(0, Vec::len) != origin_blocks {
        return Err(KdfError::InvalidFormat(
            "origin block hash count mismatch".into(),
        ));
    }
    for s in h
        .storage_rows_xxh128
        .iter()
        .chain(h.images_xxh128.iter())
        .chain(h.descriptor_blocks_xxh128.iter().flatten())
        .chain(h.origins_xxh128.iter().flatten())
    {
        parse_hash(s)?;
    }
    let mut sections = vec![parse_hash(&h.metadata_xxh128)?];
    if let Some(v) = &h.images_xxh128 {
        sections.push(parse_hash(v)?);
    }
    if let Some(values) = &h.origins_xxh128 {
        for v in values {
            sections.push(parse_hash(v)?);
        }
    }
    if let Some(v) = &h.storage_rows_xxh128 {
        sections.push(parse_hash(v)?);
    }
    if let Some(values) = &h.descriptor_blocks_xxh128 {
        for v in values {
            sections.push(parse_hash(v)?);
        }
    }
    for v in h.chunks_xxh128.iter().flatten() {
        sections.push(parse_hash(v)?);
    }
    let mut raw = Vec::with_capacity(sections.len() * 16);
    for v in sections {
        raw.extend_from_slice(&v.to_be_bytes());
    }
    if hash_string(xxh3_128(&raw)) != h.content_xxh128 {
        return Err(KdfError::Integrity(
            "whole-file digest composition mismatch".into(),
        ));
    }
    Ok(())
}

fn expected_entries<S: KdfScalar>(
    m: &Metadata,
    image_count: Option<usize>,
) -> Result<HashSet<String>, KdfError> {
    let mut owned = HashSet::from([
        "metadata.json.zst".to_string(),
        "content_hash.json.zst".to_string(),
    ]);
    if m.feature_source == "sift_files" {
        let count =
            image_count.ok_or_else(|| KdfError::InvalidFormat("missing image count".into()))?;
        owned.insert("images/metadata.json.zst".into());
        owned.insert("images/names.json.zst".into());
        owned.insert(format!("images/feature_tool_hashes.{count}.uint128.zst"));
        owned.insert(format!("images/sift_content_hashes.{count}.uint128.zst"));
        let q = m.origin_block_rows.expect("validated") as usize;
        for b in 0..(m.feature_count as usize).div_ceil(q) {
            let r = q.min(m.feature_count as usize - b * q);
            owned.insert(format!("origins/{b}/image_indexes.{r}.uint32.zst"));
            owned.insert(format!("origins/{b}/image_feature_indexes.{r}.uint32.zst"));
        }
    }
    if m.descriptor_storage == "shared" {
        owned.insert(format!(
            "features/storage_rows.{}.uint32.zst",
            m.feature_count
        ));
        let q = m.descriptor_block_rows.expect("validated") as usize;
        for b in 0..(m.feature_count as usize).div_ceil(q) {
            let r = q.min(m.feature_count as usize - b * q);
            owned.insert(format!(
                "features/blocks/{b}/vectors.{r}.{}.{}.zst",
                m.dimension,
                S::TYPE_NAME
            ));
        }
    }
    for (ti, tree) in m.trees.iter().enumerate() {
        for (ci, chunk) in tree.chunks.iter().enumerate() {
            let mc = chunk.node_count;
            let p = chunk.feature_count;
            let prefix = format!("trees/{ti}/chunks/{ci}");
            owned.insert(format!("{prefix}/nodes.{NODE_COLUMNS}.{mc}.uint32.zst"));
            owned.insert(format!("{prefix}/splits.{mc}.{}.zst", S::TYPE_NAME));
            owned.insert(format!("{prefix}/feature_ids.{p}.uint32.zst"));
            if m.descriptor_storage == "tree_local" {
                owned.insert(format!(
                    "{prefix}/vectors.{p}.{}.{}.zst",
                    m.dimension,
                    S::TYPE_NAME
                ));
            }
        }
    }
    Ok(owned)
}

fn describe_entry_difference(expected: &HashSet<String>, actual: &HashSet<String>) -> String {
    let missing = expected.difference(actual).next();
    let extra = actual.difference(expected).next();
    format!("archive entry set mismatch (missing={missing:?}, unexpected={extra:?})")
}

fn decode_nodes<S: KdfScalar>(
    m: &Metadata,
    tree: usize,
    columns: &[u32],
    splits: &[S],
    count: usize,
    features: usize,
) -> Result<Vec<DecodedNode<S>>, KdfError> {
    debug_assert_eq!(
        splits.len(),
        count,
        "caller validates the split array length"
    );
    let col = |c: usize, i: usize| columns[c * count + i];
    let mut out = Vec::with_capacity(count);
    let mut expected_start = 0usize;
    let total_nodes: usize = m.trees[tree]
        .chunks
        .iter()
        .map(|c| c.node_count as usize)
        .sum();
    for (i, &split) in splits.iter().enumerate().take(count) {
        match col(0, i) {
            0 => {
                let dim = col(2, i);
                if dim >= m.dimension as u32 || !split.is_finite() {
                    return Err(KdfError::InvalidFormat("invalid internal split".into()));
                }
                let address = |base: usize| -> Result<NodeAddress, KdfError> {
                    let a = NodeAddress {
                        chunk: col(base, i),
                        local: col(base + 1, i),
                        logical: col(base + 2, i),
                    };
                    let cm = m.trees[tree].chunks.get(a.chunk as usize).ok_or_else(|| {
                        KdfError::InvalidFormat("child chunk out of range".into())
                    })?;
                    if a.local >= cm.node_count || a.logical as usize >= total_nodes {
                        return Err(KdfError::InvalidFormat(
                            "child node/logical ID out of range".into(),
                        ));
                    }
                    Ok(a)
                };
                if col(9, i) != 0 {
                    return Err(KdfError::InvalidFormat(
                        "unused internal leaf_start is nonzero".into(),
                    ));
                }
                out.push(DecodedNode::Internal {
                    split_dimension: dim as u16,
                    split,
                    left: address(3)?,
                    right: address(6)?,
                });
            }
            1 => {
                if (2..9).any(|c| col(c, i) != 0)
                    || split.total_cmp(S::ZERO) != std::cmp::Ordering::Equal
                {
                    return Err(KdfError::InvalidFormat(
                        "unused leaf fields are nonzero".into(),
                    ));
                }
                let start = col(9, i) as usize;
                if start != expected_start {
                    return Err(KdfError::InvalidFormat(
                        "leaf starts are not contiguous in local node order".into(),
                    ));
                }
                let next = (i + 1..count)
                    .find(|&j| col(0, j) == 1)
                    .map_or(features, |j| col(9, j) as usize);
                if next <= start || next > features {
                    return Err(KdfError::InvalidFormat("invalid leaf range".into()));
                }
                expected_start = next;
                out.push(DecodedNode::Leaf {
                    start: start as u32,
                    len: (next - start) as u32,
                });
            }
            _ => return Err(KdfError::InvalidFormat("unknown node kind".into())),
        }
    }
    if expected_start != features {
        return Err(KdfError::InvalidFormat(
            "chunk feature rows are not fully owned by leaves".into(),
        ));
    }
    Ok(out)
}

fn read_bounded_json_raw(
    archive: &mut ZipArchive<std::fs::File>,
    entries: &HashMap<String, u64>,
    name: &str,
    limit: usize,
) -> Result<Vec<u8>, KdfError> {
    read_exact_or_bounded(archive, entries, name, None, limit).map(|v| v.0)
}
fn read_exact_raw(
    archive: &mut ZipArchive<std::fs::File>,
    entries: &HashMap<String, u64>,
    name: &str,
    expected: usize,
    limit: usize,
) -> Result<Vec<u8>, KdfError> {
    read_exact_or_bounded(archive, entries, name, Some(expected), limit).map(|v| v.0)
}
fn read_exact_counted(
    archive: &mut ZipArchive<std::fs::File>,
    entries: &HashMap<String, u64>,
    name: &str,
    expected: usize,
    limit: usize,
) -> Result<(Vec<u8>, u64), KdfError> {
    read_exact_or_bounded(archive, entries, name, Some(expected), limit)
}
fn read_exact_or_bounded(
    archive: &mut ZipArchive<std::fs::File>,
    entries: &HashMap<String, u64>,
    name: &str,
    expected: Option<usize>,
    limit: usize,
) -> Result<(Vec<u8>, u64), KdfError> {
    let compressed_size = *entries
        .get(name)
        .ok_or_else(|| KdfError::InvalidFormat(format!("missing entry {name}")))?;
    if compressed_size > limit as u64 {
        return Err(KdfError::ResourceLimit(format!(
            "compressed entry {name} exceeds its resource limit"
        )));
    }
    let mut entry = archive.by_name(name)?;
    let mut compressed = Vec::with_capacity(compressed_size as usize);
    entry.read_to_end(&mut compressed)?;
    let decode_limit = expected.unwrap_or(limit);
    let raw = zstd::bulk::decompress(&compressed, decode_limit)
        .map_err(|e| KdfError::InvalidFormat(format!("zstd decode failed for {name}: {e}")))?;
    if let Some(expected) = expected {
        if raw.len() != expected {
            return Err(KdfError::ShapeMismatch(format!(
                "{name}: expected {expected} bytes, got {}",
                raw.len()
            )));
        }
    }
    Ok((raw, compressed_size))
}

fn bytes_to_pod<T: bytemuck::Pod>(
    name: &str,
    raw: &[u8],
    count: usize,
) -> Result<Vec<T>, KdfError> {
    if raw.len() != count * std::mem::size_of::<T>() {
        return Err(KdfError::ShapeMismatch(format!(
            "{name}: typed length mismatch"
        )));
    }
    if raw.is_empty() {
        return Ok(Vec::new());
    }
    if let Ok(v) = bytemuck::try_cast_slice(raw) {
        Ok(v.to_vec())
    } else {
        let mut out = vec![T::zeroed(); count];
        bytemuck::cast_slice_mut(&mut out).copy_from_slice(raw);
        Ok(out)
    }
}
fn bytes_to_hashes(name: &str, raw: &[u8], count: usize) -> Result<Vec<[u8; 16]>, KdfError> {
    if raw.len() != count * 16 {
        return Err(KdfError::ShapeMismatch(format!(
            "{name}: hash array length mismatch"
        )));
    }
    Ok(raw.as_chunks::<16>().0.to_vec())
}
fn validate_permutation(rows: &[u32], n: usize, what: &str) -> Result<(), KdfError> {
    let mut seen = vec![false; n];
    for &v in rows {
        let Some(slot) = seen.get_mut(v as usize) else {
            return Err(KdfError::InvalidFormat(format!(
                "{what} contains out-of-range row"
            )));
        };
        if std::mem::replace(slot, true) {
            return Err(KdfError::InvalidFormat(format!(
                "{what} contains a duplicate row"
            )));
        }
    }
    Ok(())
}
fn hash_string(v: u128) -> String {
    format!("{v:032x}")
}
fn parse_hash(s: &str) -> Result<u128, KdfError> {
    if s.len() != 32
        || !s
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    {
        return Err(KdfError::InvalidFormat(
            "hash is not 32-character lowercase hexadecimal".into(),
        ));
    }
    u128::from_str_radix(s, 16).map_err(|_| KdfError::InvalidFormat("invalid hash".into()))
}
