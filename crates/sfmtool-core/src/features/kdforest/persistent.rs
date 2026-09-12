// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Persistence bridge and file-backed best-bin-first traversal.

use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::path::Path;

use rayon::prelude::*;
use sfmtool_kdf_format::{
    DecodedNode, KdfFile, KdfForestData, KdfNode, KdfScalar, KdfTree, NodeAddress,
};

/// Deferred far-child entry: `(Reverse(priority), tree, logical, chunk, local)`.
///
/// `Reverse` orders the max-heap by smallest priority first; the logical node ID
/// follows it so ties break on source-forest node identity rather than on where
/// repacking happened to place the child.
type QueueEntry<D> = (Reverse<D>, u32, u32, u32, u32);

use super::build::{Node, Tree};
use super::distance::ForestScalar;
use super::search::Checked;
use super::KdForestParams;
use super::{KdForest, Neighbor};
use crate::features::kdforest::{
    KdfError, KdfImageTable, KdfIoStats, KdfSiftSources, KdfWriteOptions, LazyKdForestOptions,
};

/// Per-query counters used by parity tests and performance instrumentation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LazyQueryStats {
    pub checks: u64,
    pub pushes: u64,
    pub pops: u64,
}

impl<S> KdForest<S>
where
    S: ForestScalar + KdfScalar,
{
    /// Persist this exact topology, leaf order, and original feature IDs.
    pub fn write_kdf(
        &self,
        path: &Path,
        sources: Option<&KdfSiftSources>,
        options: &KdfWriteOptions,
    ) -> Result<(), KdfError> {
        self.write_kdf_ordered(path, sources, options, None)
    }

    /// [`write_kdf`](Self::write_kdf) with an explicit shared-corpus order.
    ///
    /// `descriptor_order[r]` is the feature stored at row `r`. `None` keeps tree
    /// 0's leaf order. Exists so an ordering policy can be measured without
    /// rebuilding the forest or touching the format: the row map a reader
    /// follows is stored either way.
    pub fn write_kdf_ordered(
        &self,
        path: &Path,
        sources: Option<&KdfSiftSources>,
        options: &KdfWriteOptions,
        descriptor_order: Option<&[u32]>,
    ) -> Result<(), KdfError> {
        let trees = self
            .trees
            .iter()
            .map(|tree| KdfTree {
                nodes: tree
                    .nodes
                    .iter()
                    .map(|node| match *node {
                        Node::Internal {
                            split_dim,
                            split_val,
                            left,
                            right,
                        } => KdfNode::Internal {
                            split_dimension: split_dim,
                            split: split_val,
                            left,
                            right,
                        },
                        Node::Leaf { start, len } => KdfNode::Leaf { start, len },
                    })
                    .collect(),
                feature_ids: tree.point_ids.clone(),
            })
            .collect();
        let data = KdfForestData {
            vectors: &self.points,
            feature_count: self.n_points,
            dimension: self.dim,
            trees,
            provenance: Some(serde_json::json!({
                "builder": "sfmtool randomized kd-forest",
                "num_trees": self.params.num_trees,
                "split_dim_candidates": self.params.split_dim_candidates,
                "leaf_size": self.params.leaf_size,
                "max_leaf_checks": self.params.max_leaf_checks,
                "seed": self.params.seed,
                "descriptor_order": if descriptor_order.is_some() {
                    "explicit"
                } else {
                    "tree_0_leaf_order"
                },
            })),
            descriptor_order,
        };
        sfmtool_kdf_format::write_kdf(path, &data, sources, options)
    }
}

/// Recover build settings from a file's provenance, falling back to the
/// balanced preset for anything a writer did not record.
fn build_params_from(provenance: Option<&serde_json::Value>) -> KdForestParams {
    let mut params = KdForestParams::balanced();
    let Some(value) = provenance else {
        return params;
    };
    let get = |key: &str| value.get(key).and_then(|v| v.as_u64());
    if let Some(v) = get("num_trees") {
        params.num_trees = v as usize;
    }
    if let Some(v) = get("leaf_size") {
        params.leaf_size = v as usize;
    }
    if let Some(v) = get("split_dim_candidates") {
        params.split_dim_candidates = v as usize;
    }
    if let Some(v) = get("max_leaf_checks") {
        params.max_leaf_checks = v as usize;
    }
    if let Some(v) = get("seed") {
        params.seed = v;
    }
    params
}

impl<S> KdForest<S>
where
    S: ForestScalar + KdfScalar,
{
    /// Rebuild a full in-memory forest from a `.kdf`, without re-running a build.
    ///
    /// The third option between querying the file lazily and rebuilding from the
    /// `.sift` corpus. The file stores the exact topology, leaf order and feature
    /// IDs of the forest it was written from, so this is decompression and
    /// reassembly — no median splits, no randomization, and no dependence on the
    /// builder having stayed the same.
    ///
    /// Node identity is what makes reassembly possible: every node carries the
    /// logical ID it had in the source arena, and every child reference repeats
    /// the addressed child's logical ID, so the arena is rebuilt by placing each
    /// node at its own ID and rewriting child links to the IDs already stored.
    ///
    /// Leaf ranges are chunk-local on disk, so each chunk's feature IDs are
    /// appended to the tree's point list and its leaf starts shifted by where
    /// that chunk landed. The resulting point order is a valid leaf order but not
    /// necessarily the byte-for-byte order the original build produced; leaf
    /// membership, and therefore every query result, is identical either way.
    pub fn read_kdf(path: &Path, options: LazyKdForestOptions) -> Result<Self, KdfError> {
        let file = KdfFile::<S>::open(path, options)?;
        let n_points = file.len();
        let dim = file.dim();
        let mut points = vec![S::ZERO; n_points * dim];
        let mut have_points = false;
        let mut trees = Vec::with_capacity(file.tree_count());

        for ti in 0..file.tree_count() {
            let mut nodes = vec![Node::Leaf { start: 0, len: 0 }; file.tree_node_count(ti)];
            let mut defined = vec![false; nodes.len()];
            let mut point_ids: Vec<u32> = Vec::with_capacity(n_points);
            for ci in 0..file.chunk_count(ti) {
                let chunk = file.decoded_chunk(ti as u32, ci as u32)?;
                let base = point_ids.len() as u32;
                // Tree-local layout carries the corpus in its chunks, so the one
                // pass that reads every chunk also collects the points; the
                // shared layout keeps them elsewhere and is handled below.
                if let Some(vectors) = &chunk.vectors {
                    have_points = true;
                    for (row, &id) in chunk.feature_ids.iter().enumerate() {
                        let to = id as usize * dim;
                        points[to..to + dim].copy_from_slice(&vectors[row * dim..(row + 1) * dim]);
                    }
                }
                point_ids.extend_from_slice(&chunk.feature_ids);
                for (local, node) in chunk.nodes.iter().enumerate() {
                    let logical = chunk.logical_node_ids[local] as usize;
                    let slot = nodes.get_mut(logical).ok_or_else(|| {
                        KdfError::InvalidFormat(format!(
                            "tree {ti} logical node {logical} is out of range"
                        ))
                    })?;
                    if std::mem::replace(&mut defined[logical], true) {
                        return Err(KdfError::InvalidFormat("duplicate logical node ID".into()));
                    }
                    *slot = match *node {
                        DecodedNode::Internal {
                            split_dimension,
                            split,
                            left,
                            right,
                        } => Node::Internal {
                            split_dim: split_dimension,
                            split_val: split,
                            left: left.logical,
                            right: right.logical,
                        },
                        DecodedNode::Leaf { start, len } => Node::Leaf {
                            start: base + start,
                            len,
                        },
                    };
                }
            }
            if file.root(ti).is_some_and(|root| root.logical != 0) {
                return Err(KdfError::InvalidFormat(
                    "in-memory tree requires logical root 0".into(),
                ));
            }
            validate_loaded_tree(&nodes, &point_ids, n_points)?;
            trees.push(Tree { nodes, point_ids });
        }

        if !have_points {
            // Shared layout: one pass over the blocks, scattering each block's
            // rows to the feature IDs the row map names. Two things this avoids,
            // both measured at 9.7M descriptors. Walking feature-ID order instead
            // of storage order makes a bounded cache decode a block per
            // descriptor, because feature-ID order is the tree-0 leaf permutation
            // — twenty minutes against seconds. And reading through the
            // single-vector accessor pays a lock and a cache lookup per
            // descriptor, which cost more than rebuilding the index from
            // scratch.
            let order = file.storage_order().ok_or_else(|| {
                KdfError::InvalidFormat("shared layout without a storage row map".into())
            })?;
            let (rows, blocks) = file.descriptor_block_shape().ok_or_else(|| {
                KdfError::InvalidFormat("shared layout without a block shape".into())
            })?;
            for block in 0..blocks {
                let vectors = file.descriptor_block_vectors(block as u32)?;
                let base = block * rows;
                for (row, vector) in vectors.chunks_exact(dim).enumerate() {
                    let to = order[base + row] as usize * dim;
                    points[to..to + dim].copy_from_slice(vector);
                }
            }
        }

        Ok(Self {
            points,
            n_points,
            dim,
            // The writer records its build settings; they do not affect the
            // stored topology, only what a later query defaults its budget to.
            params: build_params_from(file.provenance()),
            trees,
        })
    }
}

/// The eager traversal omits the lazy reader's cycle checks. Validate the
/// reconstructed graph before exposing it to that unchecked traversal.
fn validate_loaded_tree<S: ForestScalar>(
    nodes: &[Node<S>],
    ids: &[u32],
    n: usize,
) -> Result<(), KdfError> {
    let mut seen = vec![false; nodes.len()];
    let mut features = vec![false; n];
    let mut stack = Vec::new();
    if !nodes.is_empty() {
        stack.push(0usize);
    }
    while let Some(at) = stack.pop() {
        let mark = seen
            .get_mut(at)
            .ok_or_else(|| KdfError::InvalidFormat("child node out of range".into()))?;
        if std::mem::replace(mark, true) {
            return Err(KdfError::InvalidFormat(
                "tree repeats a node or contains a cycle".into(),
            ));
        }
        match nodes[at] {
            Node::Internal { left, right, .. } => {
                stack.push(left as usize);
                stack.push(right as usize);
            }
            Node::Leaf { start, len } => {
                let end = (start as usize)
                    .checked_add(len as usize)
                    .ok_or_else(|| KdfError::InvalidFormat("leaf range overflow".into()))?;
                let leaf = ids
                    .get(start as usize..end)
                    .ok_or_else(|| KdfError::InvalidFormat("leaf range out of bounds".into()))?;
                for &id in leaf {
                    let mark = features
                        .get_mut(id as usize)
                        .ok_or_else(|| KdfError::InvalidFormat("feature ID out of range".into()))?;
                    if std::mem::replace(mark, true) {
                        return Err(KdfError::InvalidFormat("tree repeats a feature".into()));
                    }
                }
            }
        }
    }
    if seen.iter().any(|v| !v) || features.iter().any(|v| !v) {
        return Err(KdfError::InvalidFormat(
            "tree has unreachable nodes or missing features".into(),
        ));
    }
    Ok(())
}

/// A file-backed randomized kd-forest with a shared bounded decoded cache.
pub struct LazyKdForest<S: ForestScalar + KdfScalar> {
    file: KdfFile<S>,
    workers: rayon::ThreadPool,
    /// Start of each tree's logical node IDs in one flat index space, plus a
    /// final total. Lets one bitset cover every tree's nodes.
    tree_node_offsets: Vec<u32>,
    total_nodes: usize,
}

pub type LazyKdForestU8 = LazyKdForest<u8>;
pub type LazyKdForestF32 = LazyKdForest<f32>;

impl<S> LazyKdForest<S>
where
    S: ForestScalar + KdfScalar,
{
    pub fn open(path: &Path, options: LazyKdForestOptions) -> Result<Self, KdfError> {
        let worker_count = options.query_workers;
        let file = KdfFile::open(path, options)?;
        let workers = rayon::ThreadPoolBuilder::new()
            .num_threads(worker_count)
            .thread_name(|i| format!("kdf-query-{i}"))
            .build()
            .map_err(|e| KdfError::ResourceLimit(format!("could not create query pool: {e}")))?;
        let mut tree_node_offsets = Vec::with_capacity(file.tree_count() + 1);
        let mut total = 0usize;
        for tree in 0..file.tree_count() {
            tree_node_offsets.push(total as u32);
            total += file.tree_node_count(tree);
        }
        tree_node_offsets.push(total as u32);
        Ok(Self {
            file,
            workers,
            tree_node_offsets,
            total_nodes: total,
        })
    }

    pub fn len(&self) -> usize {
        self.file.len()
    }
    pub fn is_empty(&self) -> bool {
        self.file.is_empty()
    }
    pub fn dim(&self) -> usize {
        self.file.dim()
    }
    pub fn io_stats(&self) -> KdfIoStats {
        self.file.io_stats()
    }
    /// Zero the cumulative I/O counters, keeping resident and in-flight bytes.
    pub fn reset_io_stats(&self) {
        self.file.reset_io_stats();
    }
    pub fn image_table(&self) -> Result<Option<&KdfImageTable>, KdfError> {
        self.file.image_table()
    }
    pub fn resolve_origins(
        &self,
        ids: &[u32],
    ) -> Result<Option<Vec<super::FeatureOrigin>>, KdfError> {
        self.file.resolve_origins(ids)
    }

    pub fn search(
        &self,
        query: &[S],
        k: usize,
        max_leaf_checks: usize,
        max_dist: Option<f32>,
    ) -> Result<Vec<Neighbor>, KdfError> {
        self.search_with_stats(query, k, max_leaf_checks, max_dist)
            .map(|v| v.0)
    }

    pub fn search_with_stats(
        &self,
        query: &[S],
        k: usize,
        max_leaf_checks: usize,
        max_dist: Option<f32>,
    ) -> Result<(Vec<Neighbor>, LazyQueryStats), KdfError> {
        let mut scratch = self.new_scratch();
        let stats = self.search_into(query, k, max_leaf_checks, max_dist, &mut scratch)?;
        Ok((scratch.result.neighbors(), stats))
    }

    /// A scratch buffer sized for this forest, to reuse across many queries.
    pub fn new_scratch(&self) -> LazySearchScratch<S> {
        LazySearchScratch::new(self.len(), self.total_nodes)
    }

    /// One query, reusing `scratch`; results are left in it.
    ///
    /// The allocating entry points build a scratch per call, which is the right
    /// shape for a single query and the wrong one for a batch: a query's dedup
    /// sets, priority queue and descriptor buffer are all short-lived and all
    /// the same size every time, so a batch that reallocates them per query
    /// spends most of its time in the allocator.
    fn search_into(
        &self,
        query: &[S],
        k: usize,
        max_leaf_checks: usize,
        max_dist: Option<f32>,
        scratch: &mut LazySearchScratch<S>,
    ) -> Result<LazyQueryStats, KdfError> {
        validate_query(query, self.dim(), max_dist)?;
        let cutoff = max_dist.map(S::cutoff_sq).unwrap_or(S::MAX_DIST);
        scratch.reset(k, cutoff);
        if k == 0 || self.is_empty() {
            return Ok(LazyQueryStats::default());
        }
        let mut search = Search::<S> {
            file: &self.file,
            query,
            tree_node_offsets: &self.tree_node_offsets,
            scratch,
            stats: LazyQueryStats::default(),
        };
        for ti in 0..search.file.tree_count() {
            if let Some(root) = search.file.root(ti) {
                search.descend(ti as u32, root, S::ZERO_DIST)?;
            }
        }
        while let Some((Reverse(priority), tree, logical, chunk, local)) =
            search.scratch.queue.pop()
        {
            search.stats.pops += 1;
            if search.stats.checks >= max_leaf_checks as u64
                || priority > search.scratch.result.worst_dist()
            {
                break;
            }
            search.descend(
                tree,
                NodeAddress {
                    chunk,
                    local,
                    logical,
                },
                priority,
            )?;
        }
        Ok(search.stats)
    }

    pub fn search_batch_with_distances(
        &self,
        queries: &[S],
        n_queries: usize,
        k: usize,
        max_leaf_checks: usize,
        max_dist: Option<f32>,
    ) -> Result<(Vec<u32>, Vec<f32>), KdfError> {
        self.search_batch_with_stats(queries, n_queries, k, max_leaf_checks, max_dist)
            .map(|(i, d, _)| (i, d))
    }

    /// A batch query that also reports the batch's summed traversal counters.
    ///
    /// Read amplification — decoded chunk bytes over the vector bytes a query
    /// actually evaluated — needs both halves, and only this half is countable
    /// here: `io_stats` sees bytes moved, not how many descriptors those bytes
    /// were consulted for. The counters are summed over the batch rather than
    /// returned per query because the ratio is computed over a whole batch, and
    /// a per-query vector would allocate alongside every result row to say
    /// something no caller has asked for.
    /// [`search_batch_with_distances`](Self::search_batch_with_distances) with
    /// the queries *processed* in `order`, results still written to their own
    /// rows.
    ///
    /// Only the schedule changes, never the answers. It matters more here than
    /// for the in-memory forest: consecutive queries in descriptor-space
    /// locality order tend to reach the same tree chunks and descriptor blocks,
    /// so the cache serves them instead of the file. `order` must be a
    /// permutation of `0..n_queries`.
    pub fn search_batch_with_distances_ordered(
        &self,
        queries: &[S],
        n_queries: usize,
        k: usize,
        max_leaf_checks: usize,
        max_dist: Option<f32>,
        order: &[u32],
    ) -> Result<(Vec<u32>, Vec<f32>), KdfError> {
        if order.len() != n_queries {
            return Err(KdfError::InvalidQuery(format!(
                "order has {} entries, expected {n_queries}",
                order.len()
            )));
        }
        let mut seen = vec![false; n_queries];
        for &row in order {
            let entry = seen
                .get_mut(row as usize)
                .ok_or_else(|| KdfError::InvalidQuery("order is out of range".into()))?;
            if std::mem::replace(entry, true) {
                return Err(KdfError::InvalidQuery("order repeats a row".into()));
            }
        }
        let expected = n_queries
            .checked_mul(self.dim())
            .ok_or_else(|| KdfError::InvalidQuery("query shape overflow".into()))?;
        if queries.len() != expected {
            return Err(KdfError::InvalidQuery(format!(
                "queries contain {} scalars, expected {expected}",
                queries.len()
            )));
        }
        let width = k
            .checked_mul(n_queries)
            .ok_or_else(|| KdfError::ResourceLimit("batch output shape overflow".into()))?;
        let mut indices = vec![u32::MAX; width];
        let mut distances = vec![f32::INFINITY; width];

        if k == 0 {
            for query in queries.chunks(self.dim()) {
                validate_query(query, self.dim(), max_dist)?;
            }
            return Ok((indices, distances));
        }
        let dim = self.dim();
        self.workers.install(|| {
            indices
                .par_chunks_mut(k)
                .zip(distances.par_chunks_mut(k))
                .enumerate()
                .try_for_each_init(
                    || self.new_scratch(),
                    |scratch, (at, (out_idx, out_dist))| {
                        let row = order[at] as usize;
                        self.search_into(
                            &queries[row * dim..(row + 1) * dim],
                            k,
                            max_leaf_checks,
                            max_dist,
                            scratch,
                        )?;
                        scratch.result.write_results(out_idx, out_dist);
                        Ok::<_, KdfError>(())
                    },
                )
        })?;
        scatter_result_rows(&mut indices, &mut distances, k, &mut order.to_vec());
        Ok((indices, distances))
    }

    /// Query every stored descriptor against the forest, holding none of them.
    ///
    /// This is the self-join a whole-corpus matcher needs, and the reason it
    /// exists separately: the batch calls above take the queries as a slice, so
    /// a self-join through them would require the entire corpus in memory —
    /// exactly what a file-backed index is for avoiding. Here each query is read
    /// from the file, used, and dropped.
    ///
    /// Rows are visited in stored order, so consecutive queries fall in the same
    /// descriptor block and the cache serves most of the reads. Results are
    /// written to each feature's own row, so the output is identical to a batch
    /// over the corpus in feature-ID order.
    ///
    /// Shared layout only; a tree-local file has no corpus to read queries from.
    /// Peak memory is the cache budget plus the `n * k` result table, not the
    /// corpus.
    pub fn self_join_with_distances(
        &self,
        k: usize,
        max_leaf_checks: usize,
        max_dist: Option<f32>,
    ) -> Result<(Vec<u32>, Vec<f32>), KdfError> {
        let mut order = self.file.storage_order().ok_or_else(|| {
            KdfError::InvalidQuery(
                "a self-join needs the shared descriptor layout; this file is tree-local".into(),
            )
        })?;
        let n = self.len();
        let width = n
            .checked_mul(k)
            .ok_or_else(|| KdfError::ResourceLimit("batch output shape overflow".into()))?;
        let mut indices = vec![u32::MAX; width];
        let mut distances = vec![f32::INFINITY; width];

        if let Some(v) = max_dist {
            if v.is_nan() || v < 0.0 {
                return Err(KdfError::InvalidQuery(
                    "max_dist must be nonnegative and not NaN".into(),
                ));
            }
        }
        if k == 0 {
            return Ok((indices, distances));
        }
        self.workers.install(|| {
            indices
                .par_chunks_mut(k)
                .zip(distances.par_chunks_mut(k))
                .enumerate()
                .try_for_each_init(
                    || (self.new_scratch(), Vec::with_capacity(self.dim())),
                    |(scratch, query), (at, (out_idx, out_dist))| {
                        self.file.shared_vector_into(order[at], query)?;
                        self.search_into(query, k, max_leaf_checks, max_dist, scratch)?;
                        scratch.result.write_results(out_idx, out_dist);
                        Ok::<_, KdfError>(())
                    },
                )
        })?;
        scatter_result_rows(&mut indices, &mut distances, k, &mut order);
        Ok((indices, distances))
    }

    pub fn search_batch_with_stats(
        &self,
        queries: &[S],
        n_queries: usize,
        k: usize,
        max_leaf_checks: usize,
        max_dist: Option<f32>,
    ) -> Result<(Vec<u32>, Vec<f32>, LazyQueryStats), KdfError> {
        let expected = n_queries
            .checked_mul(self.dim())
            .ok_or_else(|| KdfError::InvalidQuery("query shape overflow".into()))?;
        if queries.len() != expected {
            return Err(KdfError::InvalidQuery(format!(
                "queries contain {} scalars, expected {expected}",
                queries.len()
            )));
        }
        if let Some(v) = max_dist {
            if v.is_nan() || v < 0.0 {
                return Err(KdfError::InvalidQuery(
                    "max_dist must be nonnegative and not NaN".into(),
                ));
            }
        }
        let width = n_queries
            .checked_mul(k)
            .ok_or_else(|| KdfError::ResourceLimit("batch output shape overflow".into()))?;
        let mut indices = vec![u32::MAX; width];
        let mut distances = vec![f32::INFINITY; width];
        if k == 0 {
            for query in queries.chunks(self.dim()) {
                validate_query(query, self.dim(), max_dist)?;
            }
            return Ok((indices, distances, LazyQueryStats::default()));
        }
        let total = self.workers.install(|| {
            indices
                .par_chunks_mut(k)
                .zip(distances.par_chunks_mut(k))
                .zip(queries.par_chunks(self.dim()))
                .map_init(
                    || self.new_scratch(),
                    |scratch, ((out_idx, out_dist), query)| {
                        let stats =
                            self.search_into(query, k, max_leaf_checks, max_dist, scratch)?;
                        scratch.result.write_results(out_idx, out_dist);
                        Ok::<_, KdfError>(stats)
                    },
                )
                .try_reduce(LazyQueryStats::default, |mut sum, next| {
                    sum.checks += next.checks;
                    sum.pushes += next.pushes;
                    sum.pops += next.pops;
                    Ok(sum)
                })
        })?;
        Ok((indices, distances, total))
    }
}

/// Rows were computed in locality order; permute them in place into feature or
/// query order. This avoids an N*k temporary plus one allocation per query.
fn scatter_result_rows(indices: &mut [u32], distances: &mut [f32], k: usize, order: &mut [u32]) {
    for at in 0..order.len() {
        while order[at] as usize != at {
            let to = order[at] as usize;
            for c in 0..k {
                indices.swap(at * k + c, to * k + c);
                distances.swap(at * k + c, to * k + c);
            }
            order.swap(at, to);
        }
    }
}

fn validate_query<S: KdfScalar>(
    query: &[S],
    dim: usize,
    max_dist: Option<f32>,
) -> Result<(), KdfError> {
    if query.len() != dim {
        return Err(KdfError::InvalidQuery(format!(
            "query dimension {} does not equal {dim}",
            query.len()
        )));
    }
    if query.iter().any(|&x| !x.is_finite()) {
        return Err(KdfError::InvalidQuery(
            "query contains a non-finite coordinate".into(),
        ));
    }
    if let Some(v) = max_dist {
        if v.is_nan() || v < 0.0 {
            return Err(KdfError::InvalidQuery(
                "max_dist must be nonnegative and not NaN".into(),
            ));
        }
    }
    Ok(())
}

/// Per-worker reusable scratch for the file-backed query path.
///
/// The in-memory forest threads one of these through a batch so the priority
/// queue, dedup sets and result buffer are allocated once per worker rather than
/// once per query; this is the same idea for the lazy path, where it matters
/// more because the leaf-ID buffer is reused too.
///
/// Both dedup sets are bitsets with a touched-word list rather than hash sets:
/// they are reset per query, and an O(words touched) reset is what makes reuse
/// worth anything. `visited` is indexed by a tree's node offset plus the node's
/// logical ID, which the format guarantees is dense within a tree.
pub struct LazySearchScratch<S: ForestScalar> {
    queue: BinaryHeap<QueueEntry<S::Dist>>,
    checked: Checked,
    visited: Checked,
    result: ResultSet<S>,
    leaf_ids: Vec<u32>,
}

impl<S: ForestScalar> LazySearchScratch<S> {
    fn new(features: usize, nodes: usize) -> Self {
        Self {
            queue: BinaryHeap::new(),
            checked: Checked::new(features),
            visited: Checked::new(nodes),
            result: ResultSet::new(0, S::MAX_DIST),
            leaf_ids: Vec::new(),
        }
    }

    fn reset(&mut self, k: usize, cutoff: S::Dist) {
        self.queue.clear();
        self.checked.clear();
        self.visited.clear();
        self.result.reset(k, cutoff);
    }
}

struct Search<'a, S: ForestScalar + KdfScalar> {
    file: &'a KdfFile<S>,
    query: &'a [S],
    tree_node_offsets: &'a [u32],
    scratch: &'a mut LazySearchScratch<S>,
    stats: LazyQueryStats,
}

impl<S: ForestScalar + KdfScalar> Search<'_, S> {
    fn descend(
        &mut self,
        tree: u32,
        mut address: NodeAddress,
        priority: S::Dist,
    ) -> Result<(), KdfError> {
        loop {
            let current_chunk = address.chunk;
            // Keep one tree pin while following nodes within this chunk. Return
            // before admitting another tree or descriptor block.
            let leaf = self
                .file
                .with_tree_chunk(tree, current_chunk, |chunk| loop {
                    let local = address.local as usize;
                    if chunk.logical_node_ids.get(local) != Some(&address.logical) {
                        return Err(KdfError::InvalidFormat(
                            "child logical ID does not match addressed node".into(),
                        ));
                    }
                    let flat = self.tree_node_offsets[tree as usize] + address.logical;
                    if !self.scratch.visited.insert(flat) {
                        return Err(KdfError::InvalidFormat(format!(
                            "tree {tree} revisits logical node {}",
                            address.logical
                        )));
                    }
                    match chunk.nodes[local] {
                        DecodedNode::Internal {
                            split_dimension,
                            split,
                            left,
                            right,
                        } => {
                            let q = self.query[split_dimension as usize];
                            let (near, far) = if ForestScalar::coord_cmp(q, split)
                                == std::cmp::Ordering::Greater
                            {
                                (right, left)
                            } else {
                                (left, right)
                            };
                            let far_priority = priority + S::axis_dist_sq(q, split);
                            if far_priority <= self.scratch.result.worst_dist() {
                                self.scratch.queue.push((
                                    Reverse(far_priority),
                                    tree,
                                    far.logical,
                                    far.chunk,
                                    far.local,
                                ));
                                self.stats.pushes += 1;
                            }
                            address = near;
                            if address.chunk != current_chunk {
                                return Ok(false);
                            }
                        }
                        DecodedNode::Leaf { start, len } => {
                            if len as usize > self.file.options_max_leaf_features() {
                                return Err(KdfError::ResourceLimit(
                                    "leaf exceeds max_leaf_features".into(),
                                ));
                            }
                            self.scratch.leaf_ids.clear();
                            for (i, &id) in chunk.feature_ids
                                [start as usize..(start + len) as usize]
                                .iter()
                                .enumerate()
                            {
                                if !self.scratch.checked.insert(id) {
                                    continue;
                                }
                                self.stats.checks += 1;
                                if let Some(vectors) = &chunk.vectors {
                                    let base = (start as usize + i) * self.file.dim();
                                    let d = S::dist_sq(
                                        self.query,
                                        &vectors[base..base + self.file.dim()],
                                    );
                                    self.scratch.result.consider(id, d);
                                } else {
                                    self.scratch.leaf_ids.push(id);
                                }
                            }
                            return Ok(true);
                        }
                    }
                })?;
            if leaf {
                if !self.scratch.leaf_ids.is_empty() {
                    self.file
                        .with_shared_vectors(&self.scratch.leaf_ids, |id, vector| {
                            self.scratch
                                .result
                                .consider(id, S::dist_sq(self.query, vector));
                        })?;
                }
                return Ok(());
            }
        }
    }
}

struct ResultSet<S: ForestScalar> {
    k: usize,
    cutoff: S::Dist,
    items: Vec<(u32, S::Dist)>,
}
impl<S: ForestScalar> ResultSet<S> {
    fn new(k: usize, cutoff: S::Dist) -> Self {
        Self {
            k,
            cutoff,
            items: Vec::with_capacity(k),
        }
    }
    fn worst_dist(&self) -> S::Dist {
        if self.k != 0 && self.items.len() == self.k {
            self.items[self.k - 1].1
        } else {
            self.cutoff
        }
    }
    fn consider(&mut self, id: u32, d: S::Dist) {
        if d > self.cutoff || (self.items.len() == self.k && d >= self.items[self.k - 1].1) {
            return;
        }
        let pos = self.items.partition_point(|&(_, value)| value <= d);
        self.items.insert(pos, (id, d));
        if self.items.len() > self.k {
            self.items.pop();
        }
    }
    fn reset(&mut self, k: usize, cutoff: S::Dist) {
        self.k = k;
        self.cutoff = cutoff;
        self.items.clear();
        self.items.reserve(k);
    }
    fn write_results(&self, indices: &mut [u32], distances: &mut [f32]) {
        for (at, &(id, distance)) in self.items.iter().enumerate() {
            indices[at] = id;
            distances[at] = S::dist_sq_to_f32(distance);
        }
    }
    fn neighbors(&self) -> Vec<Neighbor> {
        self.items
            .iter()
            .map(|&(index, d)| Neighbor {
                index,
                dist_sq: S::dist_sq_to_f32(d),
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::features::kdforest::{DescriptorStorage, KdForestParams};

    fn options() -> LazyKdForestOptions {
        LazyKdForestOptions {
            cache_bytes: 16 << 10,
            max_in_flight_bytes: 16 << 10,
            max_chunk_bytes: 16 << 10,
            max_metadata_bytes: 1 << 20,
            query_workers: 2,
            ..Default::default()
        }
    }

    #[test]
    fn eager_reassembly_rejects_cycles_and_missing_features() {
        let cycle = [
            Node::Internal {
                split_dim: 0,
                split_val: 1u8,
                left: 0,
                right: 1,
            },
            Node::Leaf { start: 0, len: 1 },
        ];
        assert!(validate_loaded_tree(&cycle, &[0], 1).is_err());
        assert!(validate_loaded_tree::<u8>(&[Node::Leaf { start: 0, len: 1 }], &[0], 2).is_err());
        assert!(
            validate_loaded_tree::<u8>(&[Node::Leaf { start: 0, len: 2 }], &[0, 0], 2).is_err()
        );
    }

    #[test]
    fn shared_reads_preserve_ties_and_reject_invalid_schedules() {
        let points = vec![7u8; 32 * 4];
        let forest = KdForest::build(
            &points,
            32,
            4,
            KdForestParams {
                num_trees: 4,
                leaf_size: 8,
                ..KdForestParams::balanced()
            },
        );
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("ties.kdf");
        let order: Vec<u32> = (0..32).rev().collect();
        forest
            .write_kdf_ordered(&path, None, &KdfWriteOptions::shared(16), Some(&order))
            .unwrap();
        let lazy = LazyKdForestU8::open(&path, LazyKdForestOptions::default()).unwrap();
        let queries = vec![7u8; 3 * 4];
        let expected = forest.search_batch_with_distances(&queries, 3, 5, 128, None);
        assert_eq!(
            lazy.search_batch_with_distances(&queries, 3, 5, 128, None)
                .unwrap(),
            expected
        );
        assert_eq!(
            lazy.search_batch_with_distances_ordered(&queries, 3, 5, 128, None, &[2, 0, 1])
                .unwrap(),
            expected
        );
        for order in [&[0, 0, 2][..], &[0, 1, 3][..]] {
            assert!(lazy
                .search_batch_with_distances_ordered(&queries, 3, 5, 128, None, order)
                .is_err());
        }
    }

    #[test]
    fn both_u8_layouts_match_eager_results_and_checks() {
        let dim = 7;
        let n = 73;
        let points: Vec<u8> = (0..n * dim)
            .map(|i| ((i * 37 + i / 5) % 251) as u8)
            .collect();
        let forest = KdForest::build(
            &points,
            n,
            dim,
            KdForestParams {
                num_trees: 4,
                leaf_size: 5,
                seed: 77,
                ..KdForestParams::balanced()
            },
        );
        let queries: Vec<u8> = (0..11 * dim).map(|i| ((i * 19 + 3) % 255) as u8).collect();
        for storage in [
            DescriptorStorage::TreeLocal,
            DescriptorStorage::Shared {
                target_descriptor_block_bytes: 19,
            },
        ] {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("forest.kdf");
            forest
                .write_kdf(
                    &path,
                    None,
                    &KdfWriteOptions {
                        descriptor_storage: storage,
                        target_chunk_bytes: 240,
                        compression_level: 1,
                        origin_block_rows: 4,
                    },
                )
                .unwrap();
            let lazy = LazyKdForestU8::open(&path, options()).unwrap();
            for (budget, query) in [0usize, 1, 7, 31, 1000]
                .into_iter()
                .flat_map(|b| queries.chunks(dim).map(move |q| (b, q)))
            {
                let expected = forest.search(query, 4, budget, Some(300.0));
                let (got, lazy_stats) = lazy
                    .search_with_stats(query, 4, budget, Some(300.0))
                    .unwrap();
                assert_eq!(got, expected, "storage={storage:?}, budget={budget}");
                let mut scratch = crate::features::kdforest::search::SearchScratch::new(n);
                let mut eager_stats = crate::features::kdforest::search::QueryStats::default();
                forest.run_query(
                    query,
                    4,
                    budget,
                    Some(300.0),
                    &mut scratch,
                    &mut eager_stats,
                );
                assert_eq!(
                    lazy_stats.checks, eager_stats.checks,
                    "storage={storage:?}, budget={budget}"
                );
            }
            let expected = forest.search_batch_with_distances(&queries, 11, 3, 30, None);
            let got = lazy
                .search_batch_with_distances(&queries, 11, 3, 30, None)
                .unwrap();
            assert_eq!(got, expected);
            let order = [3, 0, 9, 2, 10, 1, 4, 8, 5, 7, 6];
            assert_eq!(
                lazy.search_batch_with_distances_ordered(&queries, 11, 3, 30, None, &order)
                    .unwrap(),
                expected
            );
            assert_eq!(
                lazy.search_batch_with_distances(&queries, 11, 0, 30, None)
                    .unwrap(),
                (Vec::new(), Vec::new())
            );
            if matches!(storage, DescriptorStorage::Shared { .. }) {
                for k in [0, 3, 80] {
                    assert_eq!(
                        lazy.self_join_with_distances(k, 30, None).unwrap(),
                        forest.search_batch_with_distances(&points, n, k, 30, None)
                    );
                }
            }
        }
    }

    /// A forest reloaded from a file answers exactly as the one written did.
    ///
    /// This is what makes the file an index rather than a cache of one: the
    /// topology, leaf membership and feature IDs all survive the round trip, so
    /// no rebuild is needed and no randomization has to be reproduced. Checked in
    /// both layouts, because tree-local carries the corpus in its chunks while
    /// shared keeps it in one table and the reload paths differ.
    #[test]
    fn a_forest_reloaded_from_a_file_answers_identically() {
        let dim = 7;
        let n = 200;
        let points: Vec<u8> = (0..n * dim)
            .map(|i| ((i * 31 + i / 3) % 251) as u8)
            .collect();
        let forest = KdForest::build(
            &points,
            n,
            dim,
            KdForestParams {
                num_trees: 3,
                leaf_size: 8,
                seed: 5,
                ..KdForestParams::balanced()
            },
        );
        let queries: Vec<u8> = (0..9 * dim).map(|i| ((i * 17 + 5) % 255) as u8).collect();

        for storage in [
            DescriptorStorage::TreeLocal,
            DescriptorStorage::Shared {
                target_descriptor_block_bytes: 21,
            },
        ] {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("round.kdf");
            forest
                .write_kdf(
                    &path,
                    None,
                    &KdfWriteOptions {
                        descriptor_storage: storage,
                        target_chunk_bytes: 300,
                        compression_level: 1,
                        origin_block_rows: 8,
                    },
                )
                .unwrap();
            let reloaded = KdForest::<u8>::read_kdf(&path, LazyKdForestOptions::default()).unwrap();

            assert_eq!(reloaded.len(), forest.len(), "storage={storage:?}");
            assert_eq!(reloaded.dim(), forest.dim());
            assert_eq!(
                reloaded.params().num_trees,
                3,
                "params came from provenance"
            );
            assert_eq!(reloaded.params().leaf_size, 8);
            for (budget, query) in [0usize, 3, 40, 500]
                .into_iter()
                .flat_map(|b| queries.chunks(dim).map(move |q| (b, q)))
            {
                assert_eq!(
                    reloaded.search(query, 4, budget, None),
                    forest.search(query, 4, budget, None),
                    "storage={storage:?}, budget={budget}"
                );
            }
            // The corpus itself must survive, not merely the topology.
            let batch = reloaded.search_batch_with_distances(&points, n, 1, 200, None);
            let want = forest.search_batch_with_distances(&points, n, 1, 200, None);
            assert_eq!(batch, want, "storage={storage:?}");
        }
    }

    #[test]
    fn f32_signed_zero_and_cutoff_match_eager() {
        let points = vec![-0.0f32, 0.0, 1.0, 1.0, -1.0, -1.0, 2.0, 2.0];
        let forest = KdForest::build(
            &points,
            4,
            2,
            KdForestParams {
                num_trees: 2,
                leaf_size: 1,
                ..KdForestParams::balanced()
            },
        );
        for storage in [
            DescriptorStorage::TreeLocal,
            DescriptorStorage::Shared {
                target_descriptor_block_bytes: 8,
            },
        ] {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("float.kdf");
            forest
                .write_kdf(
                    &path,
                    None,
                    &KdfWriteOptions {
                        descriptor_storage: storage,
                        target_chunk_bytes: 100,
                        compression_level: 1,
                        origin_block_rows: 4,
                    },
                )
                .unwrap();
            let lazy = LazyKdForestF32::open(&path, options()).unwrap();
            for q in [[0.0, 0.0], [-0.0, 0.0], [0.5, 0.5]] {
                assert_eq!(
                    lazy.search(&q, 3, 20, Some(f32::INFINITY)).unwrap(),
                    forest.search(&q, 3, 20, Some(f32::INFINITY))
                );
            }
            assert!(lazy.search(&[f32::NAN, 0.0], 1, 1, None).is_err());
            assert!(lazy.search(&[0.0, 0.0], 1, 1, Some(-1.0)).is_err());
        }
    }

    #[test]
    fn concurrent_small_cache_eviction_completes_with_parity() {
        let dim = 16;
        let n = 160;
        let points: Vec<u8> = (0..n * dim)
            .map(|i| ((i * 43 + i / 11) % 256) as u8)
            .collect();
        let forest = KdForest::build(
            &points,
            n,
            dim,
            KdForestParams {
                num_trees: 4,
                leaf_size: 4,
                seed: 8,
                ..KdForestParams::balanced()
            },
        );
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("eviction.kdf");
        forest
            .write_kdf(
                &path,
                None,
                &KdfWriteOptions {
                    descriptor_storage: DescriptorStorage::Shared {
                        target_descriptor_block_bytes: 64,
                    },
                    target_chunk_bytes: 300,
                    compression_level: 1,
                    origin_block_rows: 4,
                },
            )
            .unwrap();
        let lazy = std::sync::Arc::new(
            LazyKdForestU8::open(
                &path,
                LazyKdForestOptions {
                    cache_bytes: 700,
                    max_in_flight_bytes: 700,
                    max_chunk_bytes: 700,
                    max_compressed_bytes: 1 << 20,
                    max_metadata_bytes: 1 << 20,
                    query_workers: 2,
                    ..Default::default()
                },
            )
            .unwrap(),
        );
        assert_eq!(
            lazy.io_stats().read_calls,
            0,
            "shared open must not read descriptor blocks"
        );
        let expected = forest.search(&points[32..48], 3, 40, None);
        let barrier = std::sync::Arc::new(std::sync::Barrier::new(8));
        let threads: Vec<_> = (0..8)
            .map(|_| {
                let lazy = lazy.clone();
                let barrier = barrier.clone();
                let q = points[32..48].to_vec();
                std::thread::spawn(move || {
                    barrier.wait();
                    lazy.search(&q, 3, 40, None).unwrap()
                })
            })
            .collect();
        for thread in threads {
            assert_eq!(thread.join().unwrap(), expected);
        }
        let stats = lazy.io_stats();
        assert!(stats.evictions > 0);
        assert!(stats.peak_resident_bytes <= 700);
        assert_eq!(stats.in_flight_bytes, 0);
    }
}
