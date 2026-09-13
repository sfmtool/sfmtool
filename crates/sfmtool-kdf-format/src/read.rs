// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashMap, HashSet};
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;
use std::sync::{Arc, Mutex, OnceLock};

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
    pub fn has_feature_geometry(&self) -> bool {
        self.geometry_corpus.is_some()
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
        let expected = &self.hashes.chunks_xxh128[tree as usize][chunk as usize];
        if hash_string(xxh3_128(&raw)) != *expected {
            return Err(KdfError::Integrity(format!(
                "tree {tree} chunk {chunk} hash mismatch"
            )));
        }
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
                if hash_string(xxh3_128(&raw))
                    != self.hashes.descriptor_blocks_xxh128[block as usize]
                {
                    return Err(KdfError::Integrity(format!(
                        "descriptor block {block} hash mismatch"
                    )));
                }
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
                if hash_string(xxh3_128(&raw))
                    != self
                        .hashes
                        .geometry_blocks_xxh128
                        .as_ref()
                        .expect("validated")[block as usize]
                {
                    return Err(KdfError::Integrity(format!(
                        "geometry block {block} hash mismatch"
                    )));
                }
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

fn validate_metadata<S: KdfScalar>(
    m: &Metadata,
    h: &ContentHash,
    o: &LazyKdForestOptions,
) -> Result<usize, KdfError> {
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

fn validate_hash_shape(h: &ContentHash, m: &Metadata) -> Result<(), KdfError> {
    for s in std::iter::once(&h.metadata_xxh128)
        .chain(std::iter::once(&h.content_xxh128))
        .chain(h.chunks_xxh128.iter().flatten())
    {
        parse_hash(s)?;
    }
    let descriptor_blocks = (m.feature_count as usize).div_ceil(m.descriptor_block_rows as usize);
    if h.descriptor_blocks_xxh128.len() != descriptor_blocks {
        return Err(KdfError::InvalidFormat(
            "descriptor block hash count mismatch".into(),
        ));
    }
    let sift = m.feature_source == "sift_files";
    if sift
        != (h.images_xxh128.is_some()
            && h.origins_xxh128.is_some()
            && h.geometry_blocks_xxh128.is_some())
    {
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
    if h.geometry_blocks_xxh128.as_ref().map_or(0, Vec::len)
        != if sift { descriptor_blocks } else { 0 }
    {
        return Err(KdfError::InvalidFormat(
            "geometry block hash count mismatch".into(),
        ));
    }
    for s in std::iter::once(&h.storage_rows_xxh128)
        .chain(h.images_xxh128.iter())
        .chain(h.descriptor_blocks_xxh128.iter())
        .chain(h.geometry_blocks_xxh128.iter().flatten())
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
    sections.push(parse_hash(&h.storage_rows_xxh128)?);
    for v in &h.descriptor_blocks_xxh128 {
        sections.push(parse_hash(v)?);
    }
    if let Some(values) = &h.geometry_blocks_xxh128 {
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
mod profiling {
    use super::*;

    #[cfg(windows)]
    #[test]
    fn independent_handles_keep_the_open_snapshot_after_replacement() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("snapshot.bin");
        std::fs::write(&path, b"old").unwrap();
        let mut original = std::fs::File::open(&path).unwrap();
        std::fs::rename(&path, dir.path().join("previous.bin")).unwrap();
        std::fs::write(&path, b"new").unwrap();
        let reopened = independent_read_handle(&original).unwrap();
        let mut out = [0; 3];
        read_at_exact(&reopened, &mut out, 0).unwrap();
        assert_eq!(&out, b"old");
        assert_eq!(
            original.stream_position().unwrap(),
            0,
            "reopening must not share the cursor"
        );
    }

    #[test]
    fn chunk_shapes_cannot_understate_admission_bytes() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("shape.kdf");
        crate::write_kdf(
            &path,
            &KdfForestData {
                vectors: &[0u8],
                feature_count: 1,
                dimension: 1,
                trees: vec![KdfTree {
                    nodes: vec![KdfNode::Leaf { start: 0, len: 1 }],
                    feature_ids: vec![0],
                }],
                provenance: None,
                descriptor_order: None,
            },
            None,
            &KdfWriteOptions {
                target_descriptor_block_bytes: 4,
                ..Default::default()
            },
        )
        .unwrap();
        let file = KdfFile::<u8>::open(&path, LazyKdForestOptions::default()).unwrap();
        let mut metadata: Metadata =
            serde_json::from_value(serde_json::to_value(&file.metadata).unwrap()).unwrap();
        metadata.trees[0].chunks[0].decoded_bytes -= 1;
        let error =
            validate_metadata::<u8>(&metadata, &file.hashes, &LazyKdForestOptions::default())
                .unwrap_err();
        assert!(
            matches!(error, KdfError::InvalidFormat(ref message) if message.contains("disagree with its shape"))
        );
    }

    #[test]
    fn concurrent_positional_reads_and_truncation() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("ranges.bin");
        let data: Vec<u8> = (0..65536)
            .map(|i| ((i * 17 + i / 257) % 251) as u8)
            .collect();
        std::fs::write(&path, &data).unwrap();
        let file = std::fs::File::open(&path).unwrap();
        std::thread::scope(|scope| {
            for worker in 0..8 {
                let file = &file;
                let data = &data;
                scope.spawn(move || {
                    for i in 0..1000 {
                        let offset = (worker * 7919 + i * 2311) % (data.len() - 37);
                        let mut out = [0u8; 37];
                        read_at_exact(file, &mut out, offset as u64).unwrap();
                        assert_eq!(&out, &data[offset..offset + 37]);
                    }
                });
            }
        });
        assert_eq!(
            read_at_exact(&file, &mut [0; 2], 65535).unwrap_err().kind(),
            std::io::ErrorKind::UnexpectedEof
        );
        read_at_exact(&file, &mut [], 65536).unwrap();
    }

    /// Diagnostic decomposition, not a throughput test: time calls separately
    /// on one thread, with an OS-warm file and no decoded-cache lookup.
    #[test]
    #[ignore = "set KDF_PROFILE_PATH to a u8 file; run in release mode"]
    fn profile_corpus_misses() {
        let path = std::env::var("KDF_PROFILE_PATH").expect("KDF_PROFILE_PATH");
        let file = KdfFile::<u8>::open(Path::new(&path), LazyKdForestOptions::default()).unwrap();
        let corpus = &file.corpus;
        let (rows, blocks) = file.descriptor_block_shape();
        let mut totals = [std::time::Duration::ZERO; 5];
        let samples = 20_000;
        for i in 0..samples {
            let b = (i * 7919) % blocks;
            let from = corpus.offsets[b];
            let length = (corpus.offsets[b + 1] - from) as usize;
            let declared = rows.min(file.len() - b * rows) * file.dim();
            let start = std::time::Instant::now();
            let mut frame = vec![0u8; length];
            {
                let mut handle = &corpus.file;
                handle
                    .seek(std::io::SeekFrom::Start(corpus.data_start + from))
                    .unwrap();
                handle.read_exact(&mut frame).unwrap();
            }
            totals[0] += start.elapsed();
            let start = std::time::Instant::now();
            let mut positioned = vec![0u8; length];
            read_at_exact(&corpus.file, &mut positioned, corpus.data_start + from).unwrap();
            totals[4] += start.elapsed();
            assert_eq!(positioned, frame);
            let start = std::time::Instant::now();
            let raw = zstd::bulk::decompress(&frame, declared).unwrap();
            totals[1] += start.elapsed();
            let start = std::time::Instant::now();
            let reused = decode_frame(&frame, declared).unwrap();
            totals[3] += start.elapsed();
            assert_eq!(raw, reused);
            let start = std::time::Instant::now();
            let digest = hash_string(xxh3_128(&raw));
            assert_eq!(&digest, &file.hashes.descriptor_blocks_xxh128[b]);
            std::hint::black_box(bytes_to_pod::<u8>("profile", &raw, declared).unwrap());
            totals[2] += start.elapsed();
        }
        for mode in [0, 1, 2] {
            let reader = KdfFile::<u8>::open(
                Path::new(&path),
                LazyKdForestOptions {
                    cache_bytes: 16 << 20,
                    query_workers: if mode == 0 { 1 } else { 4 },
                    ..Default::default()
                },
            )
            .unwrap();
            let start = std::time::Instant::now();
            std::thread::scope(|scope| {
                for worker in 0..4 {
                    let reader = &reader;
                    let path = &path;
                    scope.spawn(move || {
                        let private_handle = std::fs::File::open(path).unwrap();
                        for i in (worker..samples).step_by(4) {
                            let b = (i * 7919) % blocks;
                            if mode == 1 {
                                std::hint::black_box(reader.descriptor_block(b as u32).unwrap());
                            } else {
                                let declared = rows.min(reader.len() - b * rows) * reader.dim();
                                let raw = if mode == 2 {
                                    let corpus = &reader.corpus;
                                    let from = corpus.offsets[b];
                                    let mut frame =
                                        vec![0u8; (corpus.offsets[b + 1] - from) as usize];
                                    read_at_exact(
                                        &private_handle,
                                        &mut frame,
                                        corpus.data_start + from,
                                    )
                                    .unwrap();
                                    decode_frame(&frame, declared).unwrap()
                                } else {
                                    reader
                                        .read_corpus_frame(
                                            &reader.corpus,
                                            "descriptor",
                                            b as u32,
                                            declared,
                                        )
                                        .unwrap()
                                        .0
                                };
                                assert_eq!(
                                    hash_string(xxh3_128(&raw)),
                                    reader.hashes.descriptor_blocks_xxh128[b]
                                );
                                std::hint::black_box(
                                    bytes_to_pod::<u8>("profile", &raw, declared).unwrap(),
                                );
                            }
                        }
                    });
                }
            });
            eprintln!(
                "4 workers mode={mode} ns/completed-block={:.0} stats={:?}",
                start.elapsed().as_nanos() as f64 / samples as f64,
                reader.io_stats()
            );
        }
        eprintln!("samples={samples} block_rows={rows} ns/block read+allocation={:.0} zstd={:.0} hash+copy={:.0} reused-zstd={:.0} positional-read={:.0}",
            totals[0].as_nanos() as f64 / samples as f64,
            totals[1].as_nanos() as f64 / samples as f64,
            totals[2].as_nanos() as f64 / samples as f64,
            totals[3].as_nanos() as f64 / samples as f64,
            totals[4].as_nanos() as f64 / samples as f64);
    }
}
