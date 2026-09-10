// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Persistence bridge and file-backed best-bin-first traversal.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashSet};
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

use super::build::Node;
use super::distance::ForestScalar;
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
                "seed": self.params.seed,
            })),
        };
        sfmtool_kdf_format::write_kdf(path, &data, sources, options)
    }
}

/// A file-backed randomized kd-forest with a shared bounded decoded cache.
pub struct LazyKdForest<S: ForestScalar + KdfScalar> {
    file: KdfFile<S>,
    workers: rayon::ThreadPool,
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
        Ok(Self { file, workers })
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
        validate_query(query, self.dim(), max_dist)?;
        let cutoff = max_dist.map(S::cutoff_sq).unwrap_or(S::MAX_DIST);
        let mut search = Search::<S> {
            file: &self.file,
            query,
            queue: BinaryHeap::new(),
            checked: HashSet::new(),
            visited_nodes: HashSet::new(),
            result: ResultSet::new(k, cutoff),
            stats: LazyQueryStats::default(),
        };
        if k == 0 || self.is_empty() {
            return Ok((Vec::new(), search.stats));
        }
        for ti in 0..self.file.tree_count() {
            if let Some(root) = self.file.root(ti) {
                search.descend(ti as u32, root, S::ZERO_DIST)?;
            }
        }
        while let Some((Reverse(priority), tree, logical, chunk, local)) = search.queue.pop() {
            search.stats.pops += 1;
            if search.stats.checks >= max_leaf_checks as u64
                || priority > search.result.worst_dist()
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
        Ok((search.result.into_neighbors(), search.stats))
    }

    pub fn search_batch_with_distances(
        &self,
        queries: &[S],
        n_queries: usize,
        k: usize,
        max_leaf_checks: usize,
        max_dist: Option<f32>,
    ) -> Result<(Vec<u32>, Vec<f32>), KdfError> {
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
        let rows: Vec<Result<Vec<Neighbor>, KdfError>> = self.workers.install(|| {
            queries
                .par_chunks(self.dim())
                .map(|q| self.search(q, k, max_leaf_checks, max_dist))
                .collect()
        });
        let mut indices =
            vec![
                u32::MAX;
                n_queries
                    .checked_mul(k)
                    .ok_or_else(|| KdfError::ResourceLimit("batch output shape overflow".into()))?
            ];
        let mut distances = vec![f32::INFINITY; indices.len()];
        for (r, row) in rows.into_iter().enumerate() {
            for (c, neighbor) in row?.into_iter().enumerate() {
                indices[r * k + c] = neighbor.index;
                distances[r * k + c] = neighbor.dist_sq;
            }
        }
        Ok((indices, distances))
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

struct Search<'a, S: ForestScalar + KdfScalar> {
    file: &'a KdfFile<S>,
    query: &'a [S],
    queue: BinaryHeap<QueueEntry<S::Dist>>,
    checked: HashSet<u32>,
    visited_nodes: HashSet<(u32, u32)>,
    result: ResultSet<S>,
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
            if !self.visited_nodes.insert((tree, address.logical)) {
                return Err(KdfError::InvalidFormat(format!(
                    "tree {tree} revisits logical node {}",
                    address.logical
                )));
            }
            match self.file.node(tree, address)? {
                DecodedNode::Internal {
                    split_dimension,
                    split,
                    left,
                    right,
                } => {
                    let q = self.query[split_dimension as usize];
                    let (near, far) =
                        if ForestScalar::coord_cmp(q, split) == std::cmp::Ordering::Greater {
                            (right, left)
                        } else {
                            (left, right)
                        };
                    let far_priority = priority + S::axis_dist_sq(q, split);
                    if far_priority <= self.result.worst_dist() {
                        self.queue.push((
                            Reverse(far_priority),
                            tree,
                            far.logical,
                            far.chunk,
                            far.local,
                        ));
                        self.stats.pushes += 1;
                    }
                    address = near;
                }
                DecodedNode::Leaf { .. } => {
                    let leaf = self.file.leaf(tree, address)?;
                    if let Some(vectors) = leaf.vectors {
                        for (i, id) in leaf.feature_ids.into_iter().enumerate() {
                            if self.checked.insert(id) {
                                self.stats.checks += 1;
                                let d = S::dist_sq(
                                    self.query,
                                    &vectors[i * self.file.dim()..(i + 1) * self.file.dim()],
                                );
                                self.result.consider(id, d);
                            }
                        }
                    } else {
                        // `leaf` copied the IDs and released its tree pin before
                        // descriptor-cache admission, preventing pin cycles.
                        for id in leaf.feature_ids {
                            if self.checked.insert(id) {
                                self.stats.checks += 1;
                                let vector = self.file.shared_vector(id)?;
                                let d = S::dist_sq(self.query, &vector);
                                self.result.consider(id, d);
                            }
                        }
                    }
                    return Ok(());
                }
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
    fn into_neighbors(self) -> Vec<Neighbor> {
        self.items
            .into_iter()
            .map(|(index, d)| Neighbor {
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
