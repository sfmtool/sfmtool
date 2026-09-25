// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashMap, HashSet};
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;
use std::sync::{Arc, Mutex, OnceLock};

use rayon::prelude::*;
use xxhash_rust::xxh3::{xxh3_128, Xxh3};
use zip::ZipArchive;

use crate::cache::{Cache, CacheKey, Cached};
use crate::types::*;

#[cfg(windows)]
static NEXT_READER_SLOT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

thread_local! {
    #[cfg(windows)]
    static READER_SLOT: usize = NEXT_READER_SLOT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    // Independent frames can reuse zstd's context. One context per executing
    // thread avoids both per-miss allocation and a decoder mutex shared by workers.
    static FRAME_DECODER: std::cell::RefCell<Option<zstd::bulk::Decompressor<'static>>> = const { std::cell::RefCell::new(None) };
}

fn decode_frame(frame: &[u8], limit: usize) -> Result<Vec<u8>, std::io::Error> {
    FRAME_DECODER.with(|decoder| {
        let mut decoder = decoder.borrow_mut();
        if decoder.is_none() {
            *decoder = Some(zstd::bulk::Decompressor::new()?);
        }
        decoder
            .as_mut()
            .expect("initialized")
            .decompress(frame, limit)
    })
}

/// Reopen the same Windows file object with independent synchronous I/O state.
/// `try_clone` duplicates the handle but shares that state; reopening by path
/// could attach to a replacement file after an atomic rename.
#[cfg(windows)]
fn independent_read_handle(file: &std::fs::File) -> std::io::Result<std::fs::File> {
    use std::os::windows::io::{AsRawHandle, FromRawHandle};
    use windows_sys::Win32::Foundation::INVALID_HANDLE_VALUE;
    use windows_sys::Win32::Storage::FileSystem::{
        ReOpenFile, FILE_GENERIC_READ, FILE_SHARE_DELETE, FILE_SHARE_READ, FILE_SHARE_WRITE,
    };
    // SAFETY: file owns a live synchronous file handle throughout this call.
    let handle = unsafe {
        ReOpenFile(
            file.as_raw_handle(),
            FILE_GENERIC_READ,
            FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
            0,
        )
    };
    if handle == INVALID_HANDLE_VALUE {
        return Err(std::io::Error::last_os_error());
    }
    // SAFETY: ReOpenFile returned a new valid owned handle, transferred exactly once.
    Ok(unsafe { std::fs::File::from_raw_handle(handle) })
}

/// Read an exact byte range without sharing a seek/read critical section.
/// Windows updates the handle cursor, but every call supplies its own offset;
/// no operation on this handle relies on the cursor's previous position.
fn read_at_exact(file: &std::fs::File, mut out: &mut [u8], mut offset: u64) -> std::io::Result<()> {
    while !out.is_empty() {
        #[cfg(windows)]
        let read = std::os::windows::fs::FileExt::seek_read(file, out, offset);
        #[cfg(unix)]
        let read = std::os::unix::fs::FileExt::read_at(file, out, offset);
        #[cfg(not(any(windows, unix)))]
        let read: std::io::Result<usize> = Err(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            "positional file reads unavailable",
        ));
        match read {
            Ok(0) => return Err(std::io::ErrorKind::UnexpectedEof.into()),
            Ok(n) => {
                offset += n as u64;
                out = &mut out[n..];
            }
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e),
        }
    }
    Ok(())
}

/// Where one framed corpus lives, and how to address a block in it.
///
/// The corpus is one ZIP entry holding one independent zstd frame per block, so
/// reading a block is a seek to a recorded offset rather than a directory
/// lookup. That needs a file handle that can seek freely, separate from the one
/// the `ZipArchive` owns, and the frame boundaries, which are read at open.
struct Corpus {
    #[cfg(any(not(windows), test))]
    file: std::fs::File,
    #[cfg(windows)]
    readers: Vec<std::fs::File>,
    /// Absolute offset of the container entry's stored bytes in the file.
    data_start: u64,
    /// `blocks + 1` frame boundaries; `offsets[b]..offsets[b + 1]` is block `b`.
    offsets: Vec<u64>,
}

impl Corpus {
    fn reader(&self) -> &std::fs::File {
        #[cfg(windows)]
        {
            READER_SLOT.with(|slot| &self.readers[*slot % self.readers.len()])
        }
        #[cfg(not(windows))]
        {
            &self.file
        }
    }
}

fn locate_corpus(
    file: &std::fs::File,
    archive: &mut ZipArchive<std::fs::File>,
    entries: &HashMap<String, u64>,
    container: &str,
    offsets_name: &str,
    blocks: usize,
    options: &LazyKdForestOptions,
) -> Result<Corpus, KdfError> {
    let label = if container.starts_with("features/geometry.") {
        "geometry"
    } else {
        "descriptor"
    };
    let raw = read_exact_raw(
        archive,
        entries,
        offsets_name,
        (blocks + 1) * 8,
        options.max_compressed_bytes,
    )?;
    let offsets: Vec<u64> = bytes_to_pod(offsets_name, &raw, blocks + 1)?;
    let stored = *entries
        .get(container)
        .ok_or_else(|| KdfError::InvalidFormat(format!("{container} is missing")))?;
    let data_start = archive
        .by_name(container)?
        .data_start()
        .ok_or_else(|| KdfError::InvalidFormat(format!("{container} has no data offset")))?;
    if offsets[0] != 0 {
        return Err(KdfError::InvalidFormat(format!(
            "{label} block offsets do not start at zero"
        )));
    }
    if offsets.windows(2).any(|w| w[1] < w[0]) {
        return Err(KdfError::InvalidFormat(format!(
            "{label} block offsets are not monotonic"
        )));
    }
    if *offsets.last().expect("non-empty") != stored {
        return Err(KdfError::InvalidFormat(format!(
            "{label} block offsets end at {} but {container} stores {stored} bytes",
            offsets.last().expect("non-empty")
        )));
    }
    #[cfg(windows)]
    let readers = (0..options.query_workers)
        .map(|_| independent_read_handle(file))
        .collect::<std::io::Result<Vec<_>>>()?;
    Ok(Corpus {
        #[cfg(any(not(windows), test))]
        file: file.try_clone()?,
        #[cfg(windows)]
        readers,
        data_start,
        offsets,
    })
}

/// An open immutable `.kdf` snapshot with lazy, integrity-checked payload access.
pub struct KdfFile<S: KdfScalar> {
    archive: Mutex<ZipArchive<std::fs::File>>,
    entries: HashMap<String, u64>,
    metadata: Metadata,
    hashes: ContentHash,
    storage_rows: Vec<u32>,
    image_table: OnceLock<KdfImageTable>,
    cache: Arc<Cache<S>>,
    max_metadata_bytes: usize,
    max_compressed_bytes: usize,
    max_leaf_features: usize,
    image_count: Option<usize>,
    corpus: Corpus,
    geometry_corpus: Option<Corpus>,
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
        let mut archive = ZipArchive::new(file.try_clone()?)?;
        // `zip` indexes central-directory records by name and silently keeps
        // only one record when a malformed archive repeats a filename. Its
        // `len()` and `by_index()` therefore cannot expose duplicates to the
        // loop below. Count the raw records before trusting that unique index.
        // This also keeps duplicate rejection independent of which occurrence
        // the dependency happens to retain.
        let directory_records =
            central_directory_entry_count(file.try_clone()?, archive.central_directory_start())?;
        if directory_records != archive.len() {
            return Err(KdfError::InvalidFormat(
                "duplicate ZIP entry in central directory".into(),
            ));
        }
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
        // Before the integrity directory, not with the rest of the metadata
        // validation after it: a directory written to another version of the
        // format does not parse as this one's, and a person whose index is a
        // version behind should read that rather than a deserializer's account
        // of where the JSON stopped matching.
        check_format_and_version(&metadata)?;
        let remaining = remaining
            .checked_sub(metadata_raw.len())
            .ok_or_else(|| KdfError::ResourceLimit("metadata exceeds budget".into()))?;
        // Bounded by the metadata budget alone, not by `max_compressed_bytes`.
        // The hash directory is metadata read once at open, while
        // `max_compressed_bytes` exists to cap a per-query decode buffer; tying
        // the two meant a caller who wanted a small query buffer could not open
        // the file at all. It matters because this entry grows with the block
        // count — one digest per descriptor block — so a small block size makes
        // it large: 9.7M descriptors in 4 KiB blocks is ~303,000 digests.
        let hash_raw =
            read_bounded_json_raw(&mut archive, &entries, "content_hash.json.zst", remaining)?;
        let hashes: ContentHash = serde_json::from_slice(&hash_raw)?;
        let largest_item = validate_metadata::<S>(&metadata, &hashes, &options)?;
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

        let bytes = (metadata.feature_count as usize)
            .checked_mul(4)
            .ok_or_else(|| KdfError::ResourceLimit("address map size overflow".into()))?;
        let validation_scratch = metadata.feature_count as usize;
        if bytes
            .checked_add(validation_scratch)
            .is_none_or(|v| v > options.max_address_map_bytes)
        {
            return Err(KdfError::ResourceLimit(format!(
                "storage row map and validation scratch require more than {} bytes",
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
        if hash_string(xxh3_128(&raw)) != hashes.storage_rows_xxh128 {
            return Err(KdfError::Integrity("storage-row hash mismatch".into()));
        }
        let storage_rows: Vec<u32> = bytes_to_pod(&name, &raw, metadata.feature_count as usize)?;
        validate_permutation(
            &storage_rows,
            metadata.feature_count as usize,
            "storage row map",
        )?;

        // Descriptor and geometry frame tables are metadata: opening locates
        // them, but no frame is read until its block is requested.
        let blocks =
            (metadata.feature_count as usize).div_ceil(metadata.descriptor_block_rows as usize);
        let descriptor_container =
            corpus_entry_name::<S>(metadata.feature_count as usize, metadata.dimension as usize);
        let descriptor_offsets = block_offsets_entry_name(blocks + 1);
        let corpus = locate_corpus(
            &file,
            &mut archive,
            &entries,
            &descriptor_container,
            &descriptor_offsets,
            blocks,
            &options,
        )?;
        let geometry_corpus = if metadata.feature_source == "sift_files" {
            let geometry_container = geometry_entry_name(metadata.feature_count as usize);
            let geometry_offsets = geometry_block_offsets_entry_name(blocks + 1);
            Some(locate_corpus(
                &file,
                &mut archive,
                &entries,
                &geometry_container,
                &geometry_offsets,
                blocks,
                &options,
            )?)
        } else {
            None
        };
        let address_map_bytes = std::mem::size_of_val(storage_rows.as_slice());
        // The largest item a caller may ask for bounds the shard count: admission
        // is per shard, so a shard too small for one chunk could never admit it.
        let cache = Cache::new(
            options.cache_bytes,
            options.max_in_flight_bytes,
            address_map_bytes,
            largest_item,
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
            corpus,
            geometry_corpus,
        })
    }

    /// Number of corpus features and valid feature IDs.
    pub fn len(&self) -> usize {
        self.metadata.feature_count as usize
    }
    /// Whether the corpus contains no features.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Number of scalar coordinates in each descriptor.
    pub fn dim(&self) -> usize {
        self.metadata.dimension as usize
    }
    /// Number of trees in the forest.
    pub fn tree_count(&self) -> usize {
        self.metadata.trees.len()
    }
    /// Wire-format scalar name (`"uint8"` or `"float32"`).
    pub fn scalar_type(&self) -> &str {
        &self.metadata.scalar_type
    }
    /// Whether the file embeds SIFT centers and affine shapes.
    pub fn has_feature_geometry(&self) -> bool {
        self.geometry_corpus.is_some()
    }
    /// Configured maximum feature count accepted by [`Self::leaf`].
    pub fn options_max_leaf_features(&self) -> usize {
        self.max_leaf_features
    }
    /// Root address for a tree, or `None` for an empty tree or invalid tree index.
    pub fn root(&self, tree: usize) -> Option<NodeAddress> {
        self.metadata.trees.get(tree)?.root.map(|v| NodeAddress {
            chunk: v[0],
            local: v[1],
            logical: 0,
        })
    }
    /// Feature IDs in stored corpus order: entry `r` is the feature at row `r`.
    ///
    /// The inverse of the stored row map, and the order a self-join should visit
    /// its queries in — consecutive rows share a descriptor block, so reading
    /// them in this order is what lets a bounded cache serve a corpus it cannot
    /// hold.
    pub fn storage_order(&self) -> Vec<u32> {
        let rows = &self.storage_rows;
        let mut order = vec![0u32; rows.len()];
        for (id, &row) in rows.iter().enumerate() {
            order[row as usize] = id as u32;
        }
        order
    }

    /// The whole-file content hash the writer recorded, as 32 lowercase hex
    /// characters.
    ///
    /// Read from `content_hash.json.zst` at open and not recomputed, so it is
    /// free to ask. It covers every section of the file, so two files with the
    /// same hash hold the same corpus, trees and image table, which is what
    /// lets a file derived from an index record which index that was.
    pub fn content_xxh128(&self) -> &str {
        &self.hashes.content_xxh128
    }

    /// The writer's recorded build settings, if it left any.
    ///
    /// Provenance is free-form, so this returns the raw value rather than a typed
    /// record: the format does not constrain what a writer puts there, and a
    /// reader that rebuilds an index wants the settings without the format having
    /// to agree with the builder about their shape.
    pub fn provenance(&self) -> Option<&serde_json::Value> {
        self.metadata.provenance.as_ref()
    }

    /// Descriptor rows per block, and how many blocks there are.
    ///
    /// Together with [`storage_order`](Self::storage_order) these let a caller
    /// read the corpus a block at a time instead of a descriptor at a time.
    pub fn descriptor_block_shape(&self) -> (usize, usize) {
        let rows = self.metadata.descriptor_block_rows as usize;
        (rows, self.len().div_ceil(rows))
    }

    /// Every vector in one descriptor block, row-major.
    ///
    /// Bulk counterpart to [`vector`](Self::vector). Reading a
    /// corpus through the single-vector accessor costs a cache lookup, an `Arc`
    /// clone and a lock acquisition *per descriptor*, which on a nine-million
    /// descriptor corpus is slower than rebuilding the index from scratch. This
    /// pays those once per block.
    pub fn descriptor_block_vectors(&self, block: u32) -> Result<Vec<S>, KdfError> {
        let pin = self.descriptor_block(block)?;
        let Cached::Descriptor(vectors) = &*pin else {
            unreachable!("descriptor key yields descriptors")
        };
        Ok(vectors.clone())
    }

    /// Every SIFT geometry row corresponding to one descriptor block.
    ///
    /// Row `r` is `[[x, y], [a11, a12], [a21, a22]]`. Descriptor and geometry
    /// blocks share storage order and row boundaries, so the vectors returned by
    /// [`descriptor_block_vectors`](Self::descriptor_block_vectors) correlate
    /// positionally with these rows. Generic-vector files return `None`.
    pub fn feature_geometry_block(
        &self,
        block: u32,
    ) -> Result<Option<Vec<FeatureGeometry>>, KdfError> {
        if self.geometry_corpus.is_none() {
            return Ok(None);
        }
        let pin = self.geometry_block(block)?;
        let Cached::Geometry(rows) = &*pin else {
            unreachable!("geometry key yields geometry")
        };
        Ok(Some(rows.clone()))
    }

    /// Chunks in one tree.
    pub fn chunk_count(&self, tree: usize) -> usize {
        self.metadata.trees[tree].chunks.len()
    }

    /// A whole decoded chunk, copied out of the cache.
    ///
    /// The node-at-a-time accessors are what a query wants; rebuilding an
    /// in-memory forest wants the opposite, every node of every chunk exactly
    /// once, and going through them would decode each chunk once per node it
    /// holds. The copy is deliberate: the caller keeps the result while the pin
    /// is released, so a bulk read does not hold the cache full of chunks it has
    /// already finished with.
    pub fn decoded_chunk(&self, tree: u32, chunk: u32) -> Result<DecodedTreeChunk<S>, KdfError> {
        let pin = self.tree_chunk(tree, chunk)?;
        let Cached::Tree(decoded) = &*pin else {
            unreachable!("tree key yields a tree chunk")
        };
        Ok(decoded.clone())
    }

    /// Snapshot the handle's cache and lazy-read counters and gauges.
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

    /// Borrow one decoded tree chunk for an operation. The callback must not
    /// request another cache entry: admission can wait for this pin to drop.
    pub fn with_tree_chunk<R>(
        &self,
        tree: u32,
        chunk: u32,
        visit: impl FnOnce(&DecodedTreeChunk<S>) -> Result<R, KdfError>,
    ) -> Result<R, KdfError> {
        let pin = self.tree_chunk(tree, chunk)?;
        let Cached::Tree(chunk) = &*pin else {
            unreachable!()
        };
        visit(chunk)
    }

    /// Visit vectors in the supplied order, borrowing consecutive rows
    /// from the same block under one pin. The callback must not access the cache.
    /// Order is preserved because equal-distance search results use encounter order.
    pub fn with_vectors(
        &self,
        ids: &[u32],
        mut visit: impl FnMut(u32, &[S]),
    ) -> Result<(), KdfError> {
        let rows = &self.storage_rows;
        let q = self.metadata.descriptor_block_rows as usize;
        let row_of = |id: u32| {
            rows.get(id as usize)
                .copied()
                .map(|r| r as usize)
                .ok_or_else(|| KdfError::InvalidQuery("feature ID out of range".into()))
        };
        let mut at = 0;
        while at < ids.len() {
            let block = row_of(ids[at])? / q;
            let pin = self.descriptor_block(block as u32)?;
            let Cached::Descriptor(vectors) = &*pin else {
                unreachable!()
            };
            while at < ids.len() {
                let row = row_of(ids[at])?;
                if row / q != block {
                    break;
                }
                let base = (row % q) * self.dim();
                visit(ids[at], &vectors[base..base + self.dim()]);
                at += 1;
            }
        }
        Ok(())
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

    /// Copy a leaf's ordered members, releasing its chunk pin.
    pub fn leaf(&self, tree: u32, address: NodeAddress) -> Result<DecodedLeaf, KdfError> {
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
        Ok(DecodedLeaf {
            feature_ids: chunk.feature_ids[range].to_vec(),
        })
    }

    /// Total nodes in one tree, summed over its chunks.
    ///
    /// A caller sizing a per-query visited-node set needs this without decoding
    /// anything; the chunk directory already holds it.
    pub fn tree_node_count(&self, tree: usize) -> usize {
        self.metadata.trees[tree]
            .chunks
            .iter()
            .map(|c| c.node_count as usize)
            .sum()
    }

    /// [`vector`](Self::vector) into a caller-owned buffer.
    ///
    /// The allocating form returns a fresh `Vec` per descriptor, which a search
    /// calls once per checked candidate — so a batch of queries spends much of
    /// its time in the allocator rather than in distance work. This reuses one
    /// buffer across a whole query.
    pub fn vector_into(&self, feature_id: u32, out: &mut Vec<S>) -> Result<(), KdfError> {
        let rows = &self.storage_rows;
        let row = *rows
            .get(feature_id as usize)
            .ok_or_else(|| KdfError::InvalidQuery("feature ID out of range".into()))?
            as usize;
        let q = self.metadata.descriptor_block_rows as usize;
        let pin = self.descriptor_block((row / q) as u32)?;
        let Cached::Descriptor(vectors) = &*pin else {
            unreachable!()
        };
        let base = (row % q) * self.dim();
        out.clear();
        out.extend_from_slice(&vectors[base..base + self.dim()]);
        Ok(())
    }

    /// Copy one vector from the corpus.
    pub fn vector(&self, feature_id: u32) -> Result<Vec<S>, KdfError> {
        let rows = &self.storage_rows;
        let row = *rows
            .get(feature_id as usize)
            .ok_or_else(|| KdfError::InvalidQuery("feature ID out of range".into()))?
            as usize;
        let q = self.metadata.descriptor_block_rows as usize;
        let block = row / q;
        let within = row % q;
        let pin = self.descriptor_block(block as u32)?;
        let Cached::Descriptor(vectors) = &*pin else {
            unreachable!()
        };
        let base = within * self.dim();
        Ok(vectors[base..base + self.dim()].to_vec())
    }

    /// Copy one feature's image-space center and affine footprint.
    pub fn feature_geometry(&self, feature_id: u32) -> Result<Option<FeatureGeometry>, KdfError> {
        if self.geometry_corpus.is_none() {
            return Ok(None);
        }
        let row = *self
            .storage_rows
            .get(feature_id as usize)
            .ok_or_else(|| KdfError::InvalidQuery("feature ID out of range".into()))?
            as usize;
        let q = self.metadata.descriptor_block_rows as usize;
        let pin = self.geometry_block((row / q) as u32)?;
        let Cached::Geometry(rows) = &*pin else {
            unreachable!()
        };
        Ok(Some(rows[row % q]))
    }

    /// Resolve feature geometry in request order, including repeated IDs.
    pub fn resolve_feature_geometry(
        &self,
        feature_ids: &[u32],
    ) -> Result<Option<Vec<FeatureGeometry>>, KdfError> {
        if self.geometry_corpus.is_none() {
            return Ok(None);
        }
        feature_ids
            .iter()
            .map(|&id| Ok(self.feature_geometry(id)?.expect("geometry corpus exists")))
            .collect::<Result<Vec<_>, _>>()
            .map(Some)
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

    /// Recompute every section digest from the bytes on disk and compare.
    ///
    /// This is the whole of the file's integrity check. Opening verifies the two
    /// things it reads eagerly — the metadata and the corpus row map — and
    /// nothing else; a tree chunk, a descriptor block, a geometry block or an
    /// origin block decoded on demand is checked for shape and for the
    /// constraints its content must satisfy, but its bytes are not hashed. The
    /// cost of hashing them is proportional to the whole file, which is a price
    /// a query cannot pay and a deliberate audit can.
    ///
    /// Every block is read, so this is a full pass over the file. The corpus and
    /// geometry sections divide across the thread pool, because a block there is
    /// addressed by offset and needs no shared reader; the tree chunks and the
    /// origin blocks are ZIP entries read through one archive, and stay in order.
    pub fn verify_content(&self) -> Result<(), KdfError> {
        let metadata_digest = parse_hash(&self.hashes.metadata_xxh128)?;
        let mut sections = sfmtool_archive_io::SectionDigests::new();
        sections.push(metadata_digest);
        if self.metadata.feature_source == "sift_files" {
            // Reading the table is what checks the images section: it recomputes
            // that digest over the four entries and refuses a mismatch.
            self.image_table()?;
            sections.push(parse_hash(
                self.hashes.images_xxh128.as_deref().expect("validated"),
            )?);
            sections.push(self.check_section(
                "origins",
                self.hashes.origins_xxh128.as_deref().expect("validated"),
                self.origin_block_digests()?,
            )?);
        }
        // The row map is hashed at open, against this same digest.
        sections.push(parse_hash(&self.hashes.storage_rows_xxh128)?);
        sections.push(self.check_section(
            "descriptors",
            &self.hashes.descriptors_xxh128,
            self.corpus_block_digests(&self.corpus, "descriptor")?,
        )?);
        if let Some(corpus) = self.geometry_corpus.as_ref() {
            sections.push(self.check_section(
                "geometry",
                self.hashes.geometry_xxh128.as_deref().expect("validated"),
                self.corpus_block_digests(corpus, "geometry")?,
            )?);
        }
        sections.push(self.check_section(
            "trees",
            &self.hashes.trees_xxh128,
            self.tree_chunk_digests()?,
        )?);
        if hash_string(sections.finish()) != self.hashes.content_xxh128 {
            return Err(KdfError::Integrity("whole-file hash mismatch".into()));
        }
        Ok(())
    }

    /// Fold one section's per-item digests and compare against what is stored.
    fn check_section(&self, what: &str, stored: &str, items: Vec<u128>) -> Result<u128, KdfError> {
        let mut section = sfmtool_archive_io::SectionDigests::new();
        for digest in items {
            section.push(digest);
        }
        let digest = section.finish();
        if hash_string(digest) != stored {
            return Err(KdfError::Integrity(format!("{what} hash mismatch")));
        }
        Ok(digest)
    }

    /// Digest every block of one blocked corpus, in block order.
    fn corpus_block_digests(&self, corpus: &Corpus, label: &str) -> Result<Vec<u128>, KdfError> {
        let q = self.metadata.descriptor_block_rows as usize;
        let blocks = self.len().div_ceil(q);
        let row_bytes = if label == "geometry" {
            std::mem::size_of::<FeatureGeometry>()
        } else {
            self.dim() * std::mem::size_of::<S>()
        };
        (0..blocks)
            .into_par_iter()
            .map(|b| {
                let rows = q.min(self.len() - b * q);
                let (raw, _) = self.read_corpus_frame(corpus, label, b as u32, rows * row_bytes)?;
                Ok(xxh3_128(&raw))
            })
            .collect()
    }

    /// Digest every origin block, in block order, as the writer composed them.
    fn origin_block_digests(&self) -> Result<Vec<u128>, KdfError> {
        let q = self.metadata.origin_block_rows.expect("validated") as usize;
        let blocks = self.len().div_ceil(q);
        let mut out = Vec::with_capacity(blocks);
        for b in 0..blocks {
            let r = q.min(self.len() - b * q);
            let a_name = format!("origins/{b}/image_indexes.{r}.uint32.zst");
            let f_name = format!("origins/{b}/image_feature_indexes.{r}.uint32.zst");
            let mut archive = self.archive.lock().unwrap();
            let a_raw = read_exact_raw(
                &mut archive,
                &self.entries,
                &a_name,
                r * 4,
                self.max_compressed_bytes,
            )?;
            let f_raw = read_exact_raw(
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
            out.push(h.digest128());
        }
        Ok(out)
    }

    /// Digest every packed tree chunk, in tree order and then chunk order.
    fn tree_chunk_digests(&self) -> Result<Vec<u128>, KdfError> {
        let mut out = Vec::new();
        for (ti, tree) in self.metadata.trees.iter().enumerate() {
            for (ci, meta) in tree.chunks.iter().enumerate() {
                let m = meta.node_count as usize;
                let p = meta.feature_count as usize;
                let name = chunk_entry_name::<S>(ti, ci, m, p);
                let spans = chunk_spans::<S>(m, p);
                let mut archive = self.archive.lock().unwrap();
                let raw = read_exact_raw(
                    &mut archive,
                    &self.entries,
                    &name,
                    spans.total,
                    self.max_compressed_bytes,
                )?;
                drop(archive);
                out.push(xxh3_128(&raw));
            }
        }
        Ok(out)
    }

    fn tree_chunk(&self, tree: u32, chunk: u32) -> Result<crate::cache::CachePin<'_, S>, KdfError> {
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
        let name = chunk_entry_name::<S>(tree as usize, chunk as usize, m, p);
        let spans = chunk_spans::<S>(m, p);
        let mut archive = self.archive.lock().unwrap();
        let (raw, topology_bytes) = read_exact_counted(
            &mut archive,
            &self.entries,
            &name,
            spans.total,
            self.max_compressed_bytes,
        )?;
        drop(archive);
        let columns: Vec<u32> = bytes_to_pod(&name, &raw[..spans.nodes_end], NODE_COLUMNS * m)?;
        let splits: Vec<S> = bytes_to_pod(&name, &raw[spans.nodes_end..spans.splits_end], m)?;
        let feature_ids: Vec<u32> = bytes_to_pod(&name, &raw[spans.splits_end..spans.total], p)?;
        let nodes = decode_nodes(&self.metadata, tree as usize, &columns, &splits, m, p)?;
        if feature_ids
            .iter()
            .any(|&id| id >= self.metadata.feature_count)
        {
            return Err(KdfError::InvalidFormat("feature ID out of range".into()));
        }
        let logical_node_ids = (0..m).map(|i| columns[m + i]).collect();
        Ok((
            Cached::Tree(DecodedTreeChunk {
                logical_node_ids,
                nodes,
                feature_ids,
                decoded_bytes: meta.decoded_bytes as usize,
            }),
            topology_bytes,
        ))
    }

    /// Decode one frame out of a corpus container.
    ///
    /// A seek to a recorded offset, not a directory lookup: the container is a
    /// single ZIP entry whose stored bytes are the blocks' frames back to back.
    /// Returns the decoded bytes and the compressed length actually read, so the
    /// cache's byte accounting is unchanged from when each block was its own
    /// entry.
    fn read_corpus_frame(
        &self,
        corpus: &Corpus,
        label: &str,
        block: u32,
        declared: usize,
    ) -> Result<(Vec<u8>, u64), KdfError> {
        let b = block as usize;
        let (from, to) = match (corpus.offsets.get(b), corpus.offsets.get(b + 1)) {
            (Some(&from), Some(&to)) => (from, to),
            _ => {
                return Err(KdfError::InvalidFormat(format!(
                    "{label} block {block} is outside the corpus"
                )))
            }
        };
        let length = (to - from) as usize;
        if length > self.max_compressed_bytes {
            return Err(KdfError::ResourceLimit(format!(
                "{label} block {block} frame is {length} bytes, over the limit"
            )));
        }
        let mut frame = vec![0u8; length];
        read_at_exact(corpus.reader(), &mut frame, corpus.data_start + from)?;
        // Bound the decode by what the caller declared: a frame that expands
        // beyond its block's row count is malformed, and refusing it here keeps
        // the cache's reservation honest.
        let raw = decode_frame(&frame, declared)?;
        if raw.len() != declared {
            return Err(KdfError::ShapeMismatch(format!(
                "{label} block {block} decoded to {} bytes, expected {declared}",
                raw.len()
            )));
        }
        Ok((raw, length as u64))
    }

    fn descriptor_block(&self, block: u32) -> Result<crate::cache::CachePin<'_, S>, KdfError> {
        let q = self.metadata.descriptor_block_rows as usize;
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
                let (raw, compressed) =
                    self.read_corpus_frame(&self.corpus, "descriptor", block, declared)?;
                let values: Vec<S> = bytes_to_pod(
                    &corpus_entry_name::<S>(self.len(), self.dim()),
                    &raw,
                    r * self.dim(),
                )?;
                if values.iter().any(|&x| !x.is_finite()) {
                    return Err(KdfError::InvalidFormat(
                        "descriptor block contains non-finite vector".into(),
                    ));
                }
                Ok((Cached::Descriptor(values), compressed))
            })
    }

    fn geometry_block(&self, block: u32) -> Result<crate::cache::CachePin<'_, S>, KdfError> {
        let corpus = self
            .geometry_corpus
            .as_ref()
            .ok_or_else(|| KdfError::InvalidFormat("forest has no feature geometry".into()))?;
        let q = self.metadata.descriptor_block_rows as usize;
        let start = block as usize * q;
        if start >= self.len() {
            return Err(KdfError::InvalidQuery("geometry block out of range".into()));
        }
        let rows = q.min(self.len() - start);
        let declared = rows * std::mem::size_of::<FeatureGeometry>();
        self.cache
            .get_or_load(CacheKey::Geometry(block), declared, || {
                let (raw, compressed) =
                    self.read_corpus_frame(corpus, "geometry", block, declared)?;
                let values: Vec<FeatureGeometry> =
                    bytes_to_pod(&geometry_entry_name(self.len()), &raw, rows)?;
                if values.iter().flatten().flatten().any(|v| !v.is_finite()) {
                    return Err(KdfError::InvalidFormat(
                        "geometry block contains non-finite value".into(),
                    ));
                }
                Ok((Cached::Geometry(values), compressed))
            })
    }

    fn origin_block(&self, block: u32) -> Result<crate::cache::CachePin<'_, S>, KdfError> {
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

/// Count file records in the raw central directory.
///
/// ZIP's central file header is 46 bytes including its signature, followed by
/// variable name, extra-field and comment bytes. ZIP64 changes values inside
/// the extra field but not this framing, so no multi-gigabyte fixture is needed
/// to keep duplicate-name detection working for either form.
fn central_directory_entry_count(mut file: std::fs::File, start: u64) -> Result<usize, KdfError> {
    const CENTRAL_FILE_HEADER: u32 = 0x0201_4b50;
    file.seek(SeekFrom::Start(start))?;
    let mut count = 0usize;
    loop {
        let mut signature = [0u8; 4];
        file.read_exact(&mut signature)?;
        if u32::from_le_bytes(signature) != CENTRAL_FILE_HEADER {
            break;
        }
        let mut fixed = [0u8; 42];
        file.read_exact(&mut fixed)?;
        let name = u16::from_le_bytes([fixed[24], fixed[25]]) as i64;
        let extra = u16::from_le_bytes([fixed[26], fixed[27]]) as i64;
        let comment = u16::from_le_bytes([fixed[28], fixed[29]]) as i64;
        file.seek(SeekFrom::Current(name + extra + comment))?;
        count = count
            .checked_add(1)
            .ok_or_else(|| KdfError::ResourceLimit("ZIP entry count overflow".into()))?;
    }
    Ok(count)
}

/// Refuse a file this build does not read, before anything else is decoded.
///
/// A version this build does not write is not read either, in either direction:
/// there is one on-disk shape at a time and no translation layer. The message
/// names the remedy, because an index is derived from the `.sift` files it was
/// built from and rebuilding it is the whole fix.
fn check_format_and_version(m: &Metadata) -> Result<(), KdfError> {
    if m.format != "kdf" || m.metric != "squared_l2" {
        return Err(KdfError::InvalidFormat(
            "unsupported format or metric".into(),
        ));
    }
    if m.version != KDF_FORMAT_VERSION {
        return Err(KdfError::InvalidFormat(format!(
            "this is a version {} index and this build reads version {}; rebuild the index",
            m.version, KDF_FORMAT_VERSION
        )));
    }
    Ok(())
}

fn validate_metadata<S: KdfScalar>(
    m: &Metadata,
    h: &ContentHash,
    o: &LazyKdForestOptions,
) -> Result<usize, KdfError> {
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
    if !matches!(m.feature_source.as_str(), "none" | "sift_files") {
        return Err(KdfError::InvalidFormat("unsupported feature source".into()));
    }
    if m.feature_source == "sift_files" && (S::TYPE_NAME != "uint8" || m.dimension != 128) {
        return Err(KdfError::InvalidFormat(
            "SIFT references require unchanged 128-D uint8 descriptors".into(),
        ));
    }
    if m.descriptor_block_rows == 0 {
        return Err(KdfError::InvalidFormat(
            "descriptor_block_rows must be positive".into(),
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
    let mut largest = 0usize;
    for tree in m.trees.iter() {
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
            let scalar = std::mem::size_of::<S>() as u64;
            let node_bytes = c.node_count as u64 * (NODE_COLUMNS as u64 * 4 + scalar);
            let feature_bytes = c.feature_count as u64 * 4;
            let expected = node_bytes
                .checked_add(feature_bytes)
                .ok_or_else(|| KdfError::ResourceLimit("decoded chunk shape overflow".into()))?;
            if c.decoded_bytes != expected {
                return Err(KdfError::InvalidFormat(
                    "declared chunk bytes disagree with its shape".into(),
                ));
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
    let descriptor_rows = (m.descriptor_block_rows as usize).min(m.feature_count as usize);
    largest = largest.max(descriptor_rows * row_bytes);
    if m.feature_source == "sift_files" {
        largest = largest.max(descriptor_rows * std::mem::size_of::<FeatureGeometry>());
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
    Ok(largest)
}

/// Check the integrity directory against what the metadata says the file holds.
///
/// There is one digest per section and the set of sections is decided by the
/// feature source, so this is a presence test and a syntax test. Nothing here
/// counts digests against blocks or chunks: the directory's size does not depend
/// on the corpus.
fn validate_hash_shape(h: &ContentHash, m: &Metadata) -> Result<(), KdfError> {
    let sift = m.feature_source == "sift_files";
    if sift
        != (h.images_xxh128.is_some() && h.origins_xxh128.is_some() && h.geometry_xxh128.is_some())
    {
        return Err(KdfError::InvalidFormat(
            "source hash fields mismatch feature source".into(),
        ));
    }
    for s in [
        &h.metadata_xxh128,
        &h.content_xxh128,
        &h.storage_rows_xxh128,
        &h.descriptors_xxh128,
        &h.trees_xxh128,
    ]
    .into_iter()
    .chain(h.images_xxh128.iter())
    .chain(h.origins_xxh128.iter())
    .chain(h.geometry_xxh128.iter())
    {
        parse_hash(s)?;
    }
    // The whole-file digest is a fold over the section digests, so a directory
    // that disagrees with itself is caught here, before a byte of payload is
    // read. What it does not say is whether either matches the data; that is
    // `verify_content`'s question.
    if hash_string(compose_content_hash(h)?) != h.content_xxh128 {
        return Err(KdfError::Integrity(
            "whole-file digest composition mismatch".into(),
        ));
    }
    Ok(())
}

/// Fold an integrity directory's section digests in the order the format fixes.
fn compose_content_hash(h: &ContentHash) -> Result<u128, KdfError> {
    let mut sections = sfmtool_archive_io::SectionDigests::new();
    for s in [Some(&h.metadata_xxh128), h.images_xxh128.as_ref()]
        .into_iter()
        .chain([
            h.origins_xxh128.as_ref(),
            Some(&h.storage_rows_xxh128),
            Some(&h.descriptors_xxh128),
            h.geometry_xxh128.as_ref(),
            Some(&h.trees_xxh128),
        ])
        .flatten()
    {
        sections.push(parse_hash(s)?);
    }
    Ok(sections.finish())
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
        let blocks = (m.feature_count as usize).div_ceil(m.descriptor_block_rows as usize);
        owned.insert(geometry_entry_name(m.feature_count as usize));
        owned.insert(geometry_block_offsets_entry_name(blocks + 1));
    }
    owned.insert(format!(
        "features/storage_rows.{}.uint32.zst",
        m.feature_count
    ));
    let blocks = (m.feature_count as usize).div_ceil(m.descriptor_block_rows as usize);
    owned.insert(corpus_entry_name::<S>(
        m.feature_count as usize,
        m.dimension as usize,
    ));
    owned.insert(block_offsets_entry_name(blocks + 1));
    for (ti, tree) in m.trees.iter().enumerate() {
        for (ci, chunk) in tree.chunks.iter().enumerate() {
            owned.insert(chunk_entry_name::<S>(
                ti,
                ci,
                chunk.node_count as usize,
                chunk.feature_count as usize,
            ));
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

#[cfg(test)]
mod profiling;
