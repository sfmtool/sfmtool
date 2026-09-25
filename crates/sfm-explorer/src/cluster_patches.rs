// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The cluster-patches file of a reconstruction: the `.matches` beside its
//! `.sfmr` that holds the SIFT index's features clustered into tracks and each
//! cluster refined into a patch, where it lives, how one is opened, how one is
//! built, and what says it is still good.
//!
//! See `specs/gui/index-files.md`. The file is made from the node's SIFT index
//! ([`crate::sift_index`]) by the second half of the index-files build
//! ([`crate::index_files`]), with the defaults of the two CLI steps that make
//! one from the command line: `sfm match --cluster`'s background-floor
//! clustering, run over the index, and `sfm cluster-patches`' refinement. Like
//! the index it is a node's, opened on sight when it is there and judged
//! against the node as it stands, and its judgement also depends on the index:
//! a cluster-patches file is only as current as the index it was built from.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use ndarray::{Array1, Array2, Array3, ArrayView2, ArrayView3};
use rayon::prelude::*;
use sfmtool_core::camera::remap::{ImageU8, ImageU8Pyramid};
use sfmtool_core::features::cluster_match::{
    background_floor_clusters_lazy, BackgroundFloorParams, Clusters, LazyClusterError,
};
use sfmtool_core::features::kdforest::{KdForestParams, KdfError, LazyKdForestU8};
use sfmtool_core::patch::cluster_refine::{
    refine_cluster_patches, warp_consistency_residuals, ClusterRefineParams, FeatureGeometry,
    MemberStatus, REFERENCE_UNREFINABLE,
};
use sfmtool_core::progress::Progress;
use sfmtool_core::{progress_note, SfmrReconstruction};
use sfmtool_matches_format::{
    ClusterPatchData, ClustersData, MatchesContentHash, MatchesData, MatchesMetadata,
    WorkspaceMetadata,
};

use crate::action_log::{Actor, Kind};
use crate::index_files::{unsaved_refusal, IndexFileState, Stopped};
use crate::scene::{ReconId, SceneNode};
use crate::sift_index::{decode_xxh128, image_fingerprint, SiftIndex};
use crate::state::AppState;

#[cfg(test)]
pub(crate) mod tests;

/// What a reconstruction's cluster-patches file is called, after the `.sfmr`'s
/// own stem, beside its `-sift-index.kdf`.
pub(crate) const CLUSTER_PATCHES_FILE_SUFFIX: &str = "-cluster-patches.matches";

/// The key under `matching_options` that records the content hash of the
/// `.kdf` the clusters were made from, which is what says whether a file was
/// built from the index open beside the node.
pub(crate) const INDEX_HASH_OPTION: &str = "index_content_xxh128";

/// `sfm match --cluster`'s background rank, `--cluster-d`.
const CLUSTER_D: usize = 10;
/// `sfm match --cluster`'s radius multiplier, `--cluster-alpha`. Held as the
/// `f64` the command records, and narrowed where the kernel takes an `f32`.
const CLUSTER_ALPHA: f64 = 0.8;
/// The fewest images a cluster spans to be kept, as `sfm match --cluster`
/// keeps them.
const CLUSTER_MIN_SIZE: usize = 2;
/// The per-query budget of the self-join over a `.kdf`, which is the
/// `background_floor_clusters_kdf` binding's default: a `.kdf` stores no
/// build-time default, so the caller always states one.
const CLUSTER_MAX_LEAF_CHECKS: usize = 128;
/// `sfm cluster-patches --patch-size`: the full template edge in keypoint-frame
/// units. The kernel takes half of it.
const PATCH_SIZE: f64 = 12.0;
/// How many clusters the refinement is given at a time. The flag is read and
/// the count reported between two batches; each cluster's refinement is
/// self-contained, so the batches change the schedule and not the answers.
const REFINE_BATCH: usize = 256;
/// The zstd level every `.matches` writer here uses.
const ZSTD_LEVEL: i32 = 3;

/// One node's open cluster-patches file, and what the node makes of it.
///
/// Holds what the file says about itself and nothing of its clusters: opening
/// one reads two JSON entries of the archive, so the look on sight costs a stat
/// and a small read, as the index's does.
#[derive(Clone)]
pub(crate) struct ClusterPatches {
    /// The `.matches` it was opened from, as the tree and the wire name it.
    pub(crate) path: PathBuf,
    /// How many images the file's image table has.
    pub(crate) images: usize,
    /// How many clusters it holds.
    pub(crate) clusters: usize,
    /// How many cluster members it holds.
    pub(crate) members: usize,
    /// The file's image names, in its own order.
    image_names: Vec<String>,
    /// Whether the file has the clusters and the cluster-patches sections.
    has_patches: bool,
    /// The content hash of the index the file records being built from.
    index_hash: Option<String>,
    /// Why this file is not the node's cluster patches as they stand, or
    /// `None` when it is.
    stale: Option<String>,
    /// What the verdict above was derived against.
    judged: Judged,
}

impl ClusterPatches {
    /// Why this file will not do for the node it is open beside, or `None`.
    pub(crate) fn stale_reason(&self) -> Option<&str> {
        self.stale.as_deref()
    }
}

/// What a cluster-patches verdict depends on, so that it is derived again when
/// one of them moves and not otherwise.
#[derive(Clone, PartialEq, Eq)]
struct Judged {
    /// The version it was derived at, so a frame that pushed nothing costs one
    /// comparison.
    serial: u64,
    /// The node's image table, as [`image_fingerprint`] digests it.
    node_images: u64,
    /// The index open beside the node: its path, its content hash and whether
    /// it was current. `None` when none was open.
    index: Option<(PathBuf, String, bool)>,
}

/// Where `node`'s cluster-patches file goes: the `.sfmr`'s sibling, named
/// after its stem, beside its SIFT index. `None` for a node with no path on
/// disk.
///
/// Spelled with this platform's separators throughout, for the reason
/// [`crate::sift_index::index_path`] is.
pub(crate) fn cluster_patches_path(node: &SceneNode) -> Option<PathBuf> {
    let path = node.path.as_ref()?;
    let stem = path.file_stem()?;
    let mut name = stem.to_os_string();
    name.push(CLUSTER_PATCHES_FILE_SUFFIX);
    Some(path.with_file_name(name).components().collect())
}

/// The index facts a verdict is judged against, read off what is open.
fn index_key(index: Option<&SiftIndex>) -> Option<(PathBuf, String, bool)> {
    index.map(|index| {
        (
            index.path.clone(),
            index.forest.content_xxh128().to_string(),
            index.stale_reason().is_none(),
        )
    })
}

/// Why the file `path` describes is not the cluster patches of `recon` as it
/// stands with `index` open beside it, or `None` when it is.
///
/// The tests are in the order the sentence is chosen by: what the file is,
/// then whether its images are the node's, then whether it was made from the
/// index that is open and that index is current. The file's images are
/// compared the way an index's are, row for row in the node's order, because a
/// cluster member names an image by its row.
fn staleness(
    path: &Path,
    file: &FileFacts,
    recon: &SfmrReconstruction,
    index: Option<&SiftIndex>,
) -> Option<String> {
    let shown = path.display();
    if !file.has_patches {
        return Some(format!(
            "{shown} has no cluster patches section, so it holds clusters that were never \
             refined."
        ));
    }
    let images = &recon.image_table.images;
    if file.image_names.len() != images.len() {
        return Some(format!(
            "{shown} covers {} images and this reconstruction has {}.",
            file.image_names.len(),
            images.len()
        ));
    }
    for (row, (named, image)) in file.image_names.iter().zip(images.iter()).enumerate() {
        if named != &image.name {
            return Some(format!(
                "{shown} has {named:?} as image {row} and this reconstruction has {:?}; the \
                 file has to be over this reconstruction's own images, in its own order.",
                image.name
            ));
        }
    }
    let Some(recorded) = &file.index_hash else {
        return Some(format!(
            "{shown} does not record the SIFT index it was made from, so nothing says its \
             clusters are over this reconstruction's features."
        ));
    };
    let Some(index) = index else {
        return Some(format!(
            "No SIFT index is open beside this reconstruction, so nothing says {shown} was \
             made from its features."
        ));
    };
    if index.forest.content_xxh128() != recorded {
        return Some(format!(
            "{shown} was made from a SIFT index other than {}.",
            index.path.display()
        ));
    }
    index.stale_reason().map(|_| {
        format!(
            "{shown} was made from the SIFT index {}, which is out of date.",
            index.path.display()
        )
    })
}

/// What opening a cluster-patches file reads out of it.
struct FileFacts {
    image_names: Vec<String>,
    clusters: usize,
    members: usize,
    has_patches: bool,
    index_hash: Option<String>,
}

/// Read what the status needs from `path`: the metadata and the image names,
/// and no binary section.
fn read_facts(path: &Path) -> Result<FileFacts, String> {
    let (metadata, image_names) = sfmtool_matches_format::read_matches_image_names(path)
        .map_err(|e| format!("Cannot open {}: {e}", path.display()))?;
    Ok(FileFacts {
        image_names,
        clusters: metadata.cluster_count.unwrap_or(0) as usize,
        members: metadata.cluster_member_count.unwrap_or(0) as usize,
        has_patches: metadata.has_clusters && metadata.has_cluster_patches,
        index_hash: metadata
            .matching_options
            .get(INDEX_HASH_OPTION)
            .and_then(|value| value.as_str())
            .map(str::to_string),
    })
}

impl AppState {
    /// The cluster-patches file open beside `id`, whatever state it is in.
    pub(crate) fn cluster_patches(&self, id: ReconId) -> Option<&ClusterPatches> {
        self.cluster_patches.get(&id)?.as_ref()
    }

    /// Which of the three states `id`'s cluster-patches file is in.
    pub(crate) fn cluster_patches_state(&self, id: ReconId) -> IndexFileState {
        match self.cluster_patches(id) {
            None => IndexFileState::None,
            Some(file) if file.stale.is_some() => IndexFileState::Stale,
            Some(_) => IndexFileState::Current,
        }
    }

    /// Where `id`'s cluster-patches file goes.
    pub(crate) fn cluster_patches_path(&self, id: ReconId) -> Option<PathBuf> {
        cluster_patches_path(self.node(id)?)
    }

    /// Look for `id`'s cluster-patches file if nothing has yet, and re-derive
    /// its state when what it was judged against has moved.
    ///
    /// Called through [`Self::refresh_index_files`], after the index's own
    /// refresh, so the verdict is taken against the index as it now stands.
    /// Like the index's, it builds nothing and a miss is remembered.
    pub(crate) fn refresh_cluster_patches(&mut self, id: ReconId) {
        if self.cluster_patches.contains_key(&id) {
            self.recheck_cluster_patches(id);
            return;
        }
        let Some(path) = self.cluster_patches_path(id).filter(|p| p.is_file()) else {
            self.cluster_patches.insert(id, None);
            return;
        };
        // The viewer's row, for the reason the index's lazy open is.
        let standing = self.action_log.actor();
        self.action_log.set_actor(Actor::Viewer);
        if let Err(why) = self.open_cluster_patches(id, path) {
            self.cluster_patches.insert(id, None);
            self.action_log.fail(Kind::Bench, why);
        }
        self.action_log.set_actor(standing);
    }

    /// Open `path` as `id`'s cluster-patches file, and say so in the Action
    /// Log.
    ///
    /// **A file that opens is adopted**, current or stale, as an index is: the
    /// row says which, and why a stale one will not do.
    pub(crate) fn open_cluster_patches(
        &mut self,
        id: ReconId,
        path: PathBuf,
    ) -> Result<PathBuf, String> {
        let file = self.load_cluster_patches(id, path)?;
        let label = self.node(id).map(|n| n.label.clone()).unwrap_or_default();
        let text = match &file.stale {
            None => format!(
                "Opened the cluster patches of {label}: {} with {} clusters",
                file.path.display(),
                file.clusters
            ),
            Some(why) => format!(
                "Opened the cluster patches of {label}: {} -- it is out of date. {why}",
                file.path.display()
            ),
        };
        let path = file.path.clone();
        self.cluster_patches.insert(id, Some(file));
        self.action_log.record(Kind::Bench, text);
        Ok(path)
    }

    /// Re-derive the state of an open cluster-patches file when the node's
    /// image table or the index open beside it has moved.
    ///
    /// Reads no file: the facts the verdict needs were read at open, and the
    /// comparisons are over a few hundred names and one hash.
    pub(crate) fn recheck_cluster_patches(&mut self, id: ReconId) {
        let Some(file) = self.cluster_patches.get(&id).and_then(Option::as_ref) else {
            return;
        };
        let Some(node) = self.node(id) else {
            return;
        };
        let serial = node.history.current_version().serial.as_u64();
        let index = self.sift_index(id);
        let key = index_key(index);
        if file.judged.serial == serial && file.judged.index == key {
            return;
        }
        let node_images = image_fingerprint(node.recon());
        let judged = Judged {
            serial,
            node_images,
            index: key,
        };
        let moved =
            judged.node_images != file.judged.node_images || judged.index != file.judged.index;
        let stale = moved.then(|| {
            let facts = FileFacts {
                image_names: file.image_names.clone(),
                clusters: file.clusters,
                members: file.members,
                has_patches: file.has_patches,
                index_hash: file.index_hash.clone(),
            };
            staleness(&file.path, &facts, node.recon(), index)
        });
        let file = self
            .cluster_patches
            .get_mut(&id)
            .and_then(Option::as_mut)
            .expect("just read");
        if let Some(stale) = stale {
            file.stale = stale;
        }
        file.judged = judged;
    }

    /// Read `path` as `id`'s cluster-patches file and judge it, without
    /// installing it.
    fn load_cluster_patches(&self, id: ReconId, path: PathBuf) -> Result<ClusterPatches, String> {
        let node = self
            .node(id)
            .ok_or_else(|| "That reconstruction is no longer loaded.".to_string())?;
        let facts = read_facts(&path)?;
        let index = self.sift_index(id);
        let stale = staleness(&path, &facts, node.recon(), index);
        Ok(ClusterPatches {
            images: facts.image_names.len(),
            clusters: facts.clusters,
            members: facts.members,
            has_patches: facts.has_patches,
            index_hash: facts.index_hash,
            image_names: facts.image_names,
            stale,
            judged: Judged {
                serial: node.history.current_version().serial.as_u64(),
                node_images: image_fingerprint(node.recon()),
                index: index_key(index),
            },
            path,
        })
    }

    /// Install a cluster-patches file a background build wrote.
    ///
    /// Judged by the same function an opened file goes through. A file that
    /// cannot be read back leaves the node with none, and says so.
    pub(crate) fn install_cluster_patches(&mut self, id: ReconId, path: PathBuf) {
        match self.load_cluster_patches(id, path) {
            Ok(file) => {
                self.cluster_patches.insert(id, Some(file));
            }
            Err(why) => {
                self.cluster_patches.insert(id, None);
                self.action_log.fail(Kind::Bench, why);
            }
        }
    }

    /// Forget the cluster-patches file of a node that has left the scene.
    pub(crate) fn forget_cluster_patches(&mut self, id: ReconId) {
        self.cluster_patches.remove(&id);
    }
}

/// What the cluster half of the build needs from the node, owned.
pub(crate) struct ClusterPlan {
    /// Where the `.matches` goes.
    path: PathBuf,
    /// Every image of the node, in its own order.
    image_names: Vec<String>,
    /// Each image's `.sift` file, where it has one.
    sift_files: Vec<Option<PathBuf>>,
    /// Each image's photograph.
    photographs: Vec<PathBuf>,
    /// Each image's camera size, which stands for its dimensions when it has
    /// no `.sift` file to read them from.
    camera_dims: Vec<[u32; 2]>,
    /// The workspace the file records, its relative path measured from the
    /// file's own directory.
    workspace: WorkspaceMetadata,
}

impl ClusterPlan {
    /// The plan for `node`'s cluster-patches file.
    pub(crate) fn of(node: &SceneNode) -> Result<Self, String> {
        let path = cluster_patches_path(node).ok_or_else(|| unsaved_refusal(node))?;
        let recon = node.recon();
        let images = &recon.image_table.images;
        let mut workspace = recon.metadata.workspace.clone();
        workspace.absolute_path = recon.workspace_dir.display().to_string();
        workspace.relative_path = match recon.measured_workspace_path(&path) {
            // `os.path.relpath` spells a directory beside itself `.`, and the
            // file the CLI writes records that.
            Some(relative) if relative.is_empty() => ".".to_string(),
            Some(relative) => relative,
            None => workspace.relative_path,
        };
        Ok(Self {
            path,
            image_names: images.iter().map(|image| image.name.clone()).collect(),
            sift_files: (0..images.len())
                .map(|image| Some(recon.sift_path_for_image(image)).filter(|p| p.is_file()))
                .collect(),
            photographs: images
                .iter()
                .map(|image| recon.workspace_dir.join(&image.name))
                .collect(),
            camera_dims: images
                .iter()
                .map(|image| {
                    let camera = &recon.image_table.cameras[image.camera_index as usize];
                    [camera.width, camera.height]
                })
                .collect(),
            workspace,
        })
    }
}

/// What a finished cluster-patches build wrote.
pub(crate) struct ClusterSummary {
    pub(crate) path: PathBuf,
    pub(crate) clusters: usize,
    pub(crate) members: usize,
}

/// What one image's `.sift` metadata says, as the file records it.
struct ImageFacts {
    features: u32,
    dims: [u32; 2],
    feature_tool: [u8; 16],
    content: [u8; 16],
}

/// Make the cluster-patches file of `plan` from `forest`, the node's index
/// at `index_path`, and write it.
///
/// The two CLI steps in one: the background-floor clustering of `sfm match
/// --cluster`, run over the index with that command's defaults, and the
/// refinement of `sfm cluster-patches` with its defaults, so a file this
/// writes holds what those two steps would write from the same index. The
/// phases are `count features`, `cluster features`, `read photographs`,
/// `refine patches` and `write cluster patches`.
pub(crate) fn build(
    plan: ClusterPlan,
    forest: &LazyKdForestU8,
    index_path: &Path,
    progress: &Progress<'_>,
) -> Result<ClusterSummary, Stopped> {
    // On the 17-image seoul_bull capture the self-join and clustering are
    // 0.58 s of the 0.96 s, the refinement 0.36 s, and the metadata reads,
    // the decode and the write a few hundredths between them. The shares are
    // sixty-fourths near those, so they sum to one exactly.
    let [counting, clustering, reading, refining, writing] =
        progress.split([1.0 / 64.0, 32.0 / 64.0, 4.0 / 64.0, 24.0 / 64.0, 3.0 / 64.0]);
    let facts = image_facts(&plan, &counting)?;
    let image_starts = image_starts(&facts, forest, index_path)?;
    let clusters = cluster(forest, &image_starts, &clustering)?;
    let (detected_positions, detected_shapes) =
        member_detections(forest, &image_starts, &clusters, index_path)?;
    let pyramids = read_photographs(&plan.photographs, &reading)?;
    let refined = refine(
        &pyramids,
        &facts,
        &clusters,
        &detected_positions,
        &detected_shapes,
        &refining,
    )?;
    drop(pyramids);
    let data = matches_data(
        &plan,
        forest,
        index_path,
        &facts,
        clusters,
        detected_positions,
        detected_shapes,
        refined,
    );
    let clusters = data.metadata.cluster_count.unwrap_or(0) as usize;
    let members = data.metadata.cluster_member_count.unwrap_or(0) as usize;
    {
        let _phase = writing.phase("write cluster patches");
        sfmtool_matches_format::write_matches(&plan.path, &data, ZSTD_LEVEL)
            .map_err(|e| Stopped::Failed(format!("Cannot write {}: {e}", plan.path.display())))?;
        // The writer reports nothing of its own, so its share of the bar is
        // filled once the file is whole.
        writing.set_fraction(1.0);
    }
    Ok(ClusterSummary {
        path: plan.path,
        clusters,
        members,
    })
}

/// Each image's feature count, dimensions and identities, off its `.sift`
/// metadata, which is where `sfm match --cluster` reads them.
fn image_facts(plan: &ClusterPlan, progress: &Progress<'_>) -> Result<Vec<ImageFacts>, Stopped> {
    let phase = progress.phase("count features");
    let done = AtomicU64::new(0);
    let total = plan.sift_files.len() as u64;
    let read: Vec<Result<Option<ImageFacts>, String>> = plan
        .sift_files
        .par_iter()
        .zip(&plan.camera_dims)
        .map(|(sift, dims)| {
            if phase.is_cancelled() {
                return Ok(None);
            }
            let facts = match sift {
                // An image with no features contributes no member and still
                // takes its row, with its camera's size for its dimensions.
                None => ImageFacts {
                    features: 0,
                    dims: *dims,
                    feature_tool: [0; 16],
                    content: [0; 16],
                },
                Some(path) => {
                    let (_, metadata, hashes) = sfmtool_sift_format::read_sift_metadata(path)
                        .map_err(|e| format!("Cannot read {}: {e}", path.display()))?;
                    ImageFacts {
                        features: metadata.feature_count,
                        dims: [metadata.image_width, metadata.image_height],
                        feature_tool: decode_xxh128(&hashes.feature_tool_xxh128).unwrap_or([0; 16]),
                        content: decode_xxh128(&hashes.content_xxh128).unwrap_or([0; 16]),
                    }
                }
            };
            let finished = done.fetch_add(1, Ordering::Relaxed) + 1;
            phase.count(finished, Some(total), "image");
            Ok(Some(facts))
        })
        .collect();
    if phase.is_cancelled() {
        return Err(Stopped::Cancelled);
    }
    let mut facts = Vec::with_capacity(read.len());
    for outcome in read {
        match outcome {
            Ok(Some(value)) => facts.push(value),
            Ok(None) => return Err(Stopped::Cancelled),
            Err(why) => return Err(Stopped::Failed(why)),
        }
    }
    Ok(facts)
}

/// The CSR offsets of each image's rows in the index, which holds them image
/// by image in the node's order, each image's in `.sift` row order.
///
/// Checked against the index's own length, so an index over other features
/// than the `.sift` files hold now is refused rather than clustered under the
/// wrong image numbers.
fn image_starts(
    facts: &[ImageFacts],
    forest: &LazyKdForestU8,
    index_path: &Path,
) -> Result<Vec<u32>, Stopped> {
    let mut starts = Vec::with_capacity(facts.len() + 1);
    let mut at = 0u64;
    starts.push(0u32);
    for image in facts {
        at += u64::from(image.features);
        starts.push(u32::try_from(at).map_err(|_| {
            Stopped::Failed("The capture holds more features than a .matches file can name.".into())
        })?);
    }
    if at != forest.len() as u64 {
        return Err(Stopped::Failed(format!(
            "{} holds {} descriptors and the .sift files hold {at}, so it is not an index of \
             these features. Rebuild the index files.",
            index_path.display(),
            forest.len()
        )));
    }
    Ok(starts)
}

/// The background-floor clustering over the index, with `sfm match
/// --cluster`'s defaults.
fn cluster(
    forest: &LazyKdForestU8,
    image_starts: &[u32],
    progress: &Progress<'_>,
) -> Result<Clusters, Stopped> {
    let mut phase = progress.phase("cluster features");
    let params = BackgroundFloorParams {
        d: CLUSTER_D,
        alpha: CLUSTER_ALPHA as f32,
        min_size: CLUSTER_MIN_SIZE,
        forest: KdForestParams {
            max_leaf_checks: CLUSTER_MAX_LEAF_CHECKS,
            ..KdForestParams::accurate()
        },
    };
    let clusters = background_floor_clusters_lazy(forest, image_starts, &params, &phase).map_err(
        |e| match e {
            LazyClusterError::Kdf(KdfError::Cancelled(_)) => Stopped::Cancelled,
            e => Stopped::Failed(format!("Cannot cluster the features: {e}")),
        },
    )?;
    progress_note!(phase, "{} clusters", clusters.cluster_starts.len() - 1);
    Ok(clusters)
}

/// Every member's detected position and affine shape, read out of the index's
/// feature geometry, which holds the `.sift` values bit for bit.
fn member_detections(
    forest: &LazyKdForestU8,
    image_starts: &[u32],
    clusters: &Clusters,
    index_path: &Path,
) -> Result<(Array2<f32>, Array3<f32>), Stopped> {
    let ids: Vec<u32> = clusters
        .member_images
        .iter()
        .zip(clusters.member_features.iter())
        .map(|(&image, &feature)| image_starts[image as usize] + feature)
        .collect();
    let geometry = forest
        .resolve_feature_geometry(&ids)
        .map_err(|e| Stopped::Failed(format!("Cannot read {}: {e}", index_path.display())))?
        .ok_or_else(|| {
            Stopped::Failed(format!(
                "{} holds no keypoint geometry, so the clusters have no detections to refine \
                 from. Rebuild the index files.",
                index_path.display()
            ))
        })?;
    let members = ids.len();
    let mut positions = Array2::<f32>::zeros((members, 2));
    let mut shapes = Array3::<f32>::zeros((members, 2, 2));
    for (k, [centre, row0, row1]) in geometry.into_iter().enumerate() {
        positions[[k, 0]] = centre[0];
        positions[[k, 1]] = centre[1];
        shapes[[k, 0, 0]] = row0[0];
        shapes[[k, 0, 1]] = row0[1];
        shapes[[k, 1, 0]] = row1[0];
        shapes[[k, 1, 1]] = row1[1];
    }
    Ok((positions, shapes))
}

/// Every photograph decoded into a full pyramid, in the node's order.
///
/// **In OpenCV's channel order.** `sfm cluster-patches` reads its photographs
/// with `cv2.imread`, which hands back blue, green, red; the refinement
/// averages a score over the channels, so the order is the order of a
/// floating-point sum, and matching it keeps the two builds' arithmetic the
/// same.
fn read_photographs(
    photographs: &[PathBuf],
    progress: &Progress<'_>,
) -> Result<Vec<ImageU8Pyramid>, Stopped> {
    let phase = progress.phase("read photographs");
    let done = AtomicU64::new(0);
    let total = photographs.len() as u64;
    let read: Vec<Result<Option<ImageU8Pyramid>, String>> = photographs
        .par_iter()
        .map(|path| {
            if phase.is_cancelled() {
                return Ok(None);
            }
            let decoded = image::open(path)
                .map_err(|e| format!("Cannot read the photograph {}: {e}", path.display()))?;
            let mut bgr = decoded.to_rgb8();
            for pixel in bgr.pixels_mut() {
                pixel.0.swap(0, 2);
            }
            let (width, height) = bgr.dimensions();
            let image = ImageU8::new(width, height, 3, bgr.into_raw());
            let pyramid =
                ImageU8Pyramid::from_image(image, ImageU8Pyramid::full_levels(width, height));
            let finished = done.fetch_add(1, Ordering::Relaxed) + 1;
            phase.count(finished, Some(total), "image");
            Ok(Some(pyramid))
        })
        .collect();
    if phase.is_cancelled() {
        return Err(Stopped::Cancelled);
    }
    let mut pyramids = Vec::with_capacity(read.len());
    for outcome in read {
        match outcome {
            Ok(Some(pyramid)) => pyramids.push(pyramid),
            Ok(None) => return Err(Stopped::Cancelled),
            Err(why) => return Err(Stopped::Failed(why)),
        }
    }
    Ok(pyramids)
}

/// The refinement's answer for every member, in the member order of the
/// clusters it was given.
struct Refined {
    reference_members: Vec<u32>,
    member_status: Vec<MemberStatus>,
    member_positions: Array2<f64>,
    member_affine_shapes: Array3<f64>,
    member_zncc: Vec<f32>,
    member_shift_px: Vec<f32>,
    member_consistency_residual: Vec<f32>,
}

/// `sfm cluster-patches`' refinement with its defaults, a batch of clusters at
/// a time so a cancel lands between two batches.
///
/// The kernel reads only the feature rows its members name, so each image's
/// arrays hold the members' detections at their own rows and zeros elsewhere,
/// which is how `sfm cluster-patches` presents them: the same values at the
/// same rows as a `.sift` read, without opening a `.sift` file.
fn refine(
    pyramids: &[ImageU8Pyramid],
    facts: &[ImageFacts],
    clusters: &Clusters,
    detected_positions: &Array2<f32>,
    detected_shapes: &Array3<f32>,
    progress: &Progress<'_>,
) -> Result<Refined, Stopped> {
    let phase = progress.phase("refine patches");
    let member_images = clusters.member_images.as_slice().expect("contiguous");
    let member_features = clusters.member_features.as_slice().expect("contiguous");
    let starts = clusters.cluster_starts.as_slice().expect("contiguous");

    let mut positions: Vec<Array2<f32>> = facts
        .iter()
        .map(|image| Array2::zeros((image.features as usize, 2)))
        .collect();
    let mut shapes: Vec<Array3<f32>> = facts
        .iter()
        .map(|image| Array3::zeros((image.features as usize, 2, 2)))
        .collect();
    for (k, (&image, &feature)) in member_images.iter().zip(member_features).enumerate() {
        let (image, feature) = (image as usize, feature as usize);
        positions[image][[feature, 0]] = detected_positions[[k, 0]];
        positions[image][[feature, 1]] = detected_positions[[k, 1]];
        for r in 0..2 {
            for c in 0..2 {
                shapes[image][[feature, r, c]] = detected_shapes[[k, r, c]];
            }
        }
    }
    let features: Vec<FeatureGeometry<'_>> = positions
        .iter()
        .zip(&shapes)
        .map(|(p, a)| FeatureGeometry {
            positions_xy: ArrayView2::from(p),
            affine_shapes: ArrayView3::from(a),
        })
        .collect();

    let defaults = ClusterRefineParams::default();
    let params = ClusterRefineParams {
        radius: PATCH_SIZE / 2.0,
        ..defaults
    };

    let cluster_count = starts.len() - 1;
    let members = member_images.len();
    let mut refined = Refined {
        reference_members: vec![REFERENCE_UNREFINABLE; cluster_count],
        member_status: vec![MemberStatus::NotEvaluated; members],
        member_positions: Array2::zeros((members, 2)),
        member_affine_shapes: Array3::zeros((members, 2, 2)),
        member_zncc: vec![f32::NAN; members],
        member_shift_px: vec![f32::NAN; members],
        member_consistency_residual: Vec::new(),
    };
    let mut first = 0;
    while first < cluster_count {
        if phase.is_cancelled() {
            return Err(Stopped::Cancelled);
        }
        let last = (first + REFINE_BATCH).min(cluster_count);
        let (m0, m1) = (starts[first] as usize, starts[last] as usize);
        let local: Vec<u32> = starts[first..=last].iter().map(|s| s - m0 as u32).collect();
        let batch = refine_cluster_patches(
            pyramids,
            &features,
            &local,
            &member_images[m0..m1],
            &member_features[m0..m1],
            &params,
            None,
        );
        for (c, reference) in batch.reference_members.into_iter().enumerate() {
            refined.reference_members[first + c] = match reference {
                REFERENCE_UNREFINABLE => REFERENCE_UNREFINABLE,
                local => local + m0 as u32,
            };
        }
        refined.member_status[m0..m1].copy_from_slice(&batch.member_status);
        refined.member_zncc[m0..m1].copy_from_slice(&batch.member_zncc);
        refined.member_shift_px[m0..m1].copy_from_slice(&batch.member_shift_px);
        refined
            .member_positions
            .slice_mut(ndarray::s![m0..m1, ..])
            .assign(&batch.member_positions);
        refined
            .member_affine_shapes
            .slice_mut(ndarray::s![m0..m1, .., ..])
            .assign(&batch.member_affine_shapes);
        first = last;
        phase.count(first as u64, Some(cluster_count as u64), "cluster");
    }
    refined.member_consistency_residual = warp_consistency_residuals(
        starts,
        member_images,
        &refined.member_status,
        &refined.reference_members,
        refined.member_affine_shapes.view(),
        pyramids.len(),
    );
    Ok(refined)
}

/// The file `sfm match --cluster` and then `sfm cluster-patches` write, from
/// what the build computed.
#[allow(clippy::too_many_arguments)]
fn matches_data(
    plan: &ClusterPlan,
    forest: &LazyKdForestU8,
    index_path: &Path,
    facts: &[ImageFacts],
    clusters: Clusters,
    detected_positions: Array2<f32>,
    detected_shapes: Array3<f32>,
    refined: Refined,
) -> MatchesData {
    let cluster_count = clusters.cluster_starts.len() - 1;
    let member_count = clusters.member_images.len();
    let index_name = index_path
        .file_name()
        .map(|name| name.to_string_lossy().to_string())
        .unwrap_or_default();
    let options = serde_json::json!({
        "mode": "background-floor",
        "d": CLUSTER_D,
        "alpha": CLUSTER_ALPHA,
        "min_size": CLUSTER_MIN_SIZE,
        "max_leaf_checks": CLUSTER_MAX_LEAF_CHECKS,
        "index": index_name,
        INDEX_HASH_OPTION: forest.content_xxh128(),
    });
    let matching_options = options
        .as_object()
        .expect("an object")
        .iter()
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect();

    // The output's stage is refinement, so the members the cascade measured
    // carry its geometry and the rest keep the detection they came in with.
    let mut positions = detected_positions;
    let mut shapes = detected_shapes;
    for (k, status) in refined.member_status.iter().enumerate() {
        let measured = matches!(
            status,
            MemberStatus::Reference
                | MemberStatus::Kept
                | MemberStatus::RejectedLowZncc
                | MemberStatus::RejectedShift
        );
        if !measured {
            continue;
        }
        positions[[k, 0]] = refined.member_positions[[k, 0]] as f32;
        positions[[k, 1]] = refined.member_positions[[k, 1]] as f32;
        for r in 0..2 {
            for c in 0..2 {
                shapes[[k, r, c]] = refined.member_affine_shapes[[k, r, c]] as f32;
            }
        }
    }

    let refine_defaults = ClusterRefineParams::default();
    MatchesData {
        metadata: MatchesMetadata {
            version: sfmtool_matches_format::MATCHES_FORMAT_VERSION,
            matching_method: "cluster".into(),
            matching_tool: "sfmtool".into(),
            matching_tool_version: env!("CARGO_PKG_VERSION").into(),
            matching_options,
            workspace: plan.workspace.clone(),
            timestamp: jiff::Zoned::now()
                .strftime("%Y-%m-%dT%H:%M:%S%.6f%:z")
                .to_string(),
            image_count: plan.image_names.len() as u32,
            image_pair_count: None,
            match_count: None,
            cluster_count: Some(cluster_count as u32),
            cluster_member_count: Some(member_count as u32),
            has_two_view_geometries: false,
            has_clusters: true,
            has_cluster_patches: true,
        },
        content_hash: MatchesContentHash {
            metadata_xxh128: String::new(),
            images_xxh128: String::new(),
            image_pairs_xxh128: None,
            clusters_xxh128: None,
            cluster_patches_xxh128: None,
            two_view_geometries_xxh128: None,
            content_xxh128: String::new(),
        },
        image_names: plan.image_names.clone(),
        feature_tool_hashes: facts.iter().map(|image| image.feature_tool).collect(),
        sift_content_hashes: facts.iter().map(|image| image.content).collect(),
        feature_counts: facts.iter().map(|image| image.features).collect(),
        image_dims: Some(
            Array2::from_shape_vec(
                (facts.len(), 2),
                facts.iter().flat_map(|image| image.dims).collect(),
            )
            .expect("two per image"),
        ),
        image_pairs: None,
        clusters: Some(ClustersData {
            cluster_starts: clusters.cluster_starts,
            member_images: clusters.member_images,
            member_features: clusters.member_features,
            member_positions: Some(positions),
            member_affine_shapes: Some(shapes),
            matcher_options: options,
        }),
        cluster_patches: Some(ClusterPatchData {
            reference_members: Array1::from(refined.reference_members),
            member_status: refined
                .member_status
                .iter()
                .map(|&status| status as u8)
                .collect(),
            member_zncc: Array1::from(refined.member_zncc),
            member_shift_px: Array1::from(refined.member_shift_px),
            member_consistency_residual: Array1::from(refined.member_consistency_residual),
            refine_options: serde_json::json!({
                "patch_size": PATCH_SIZE,
                "resolution": refine_defaults.resolution,
                "min_zncc": refine_defaults.min_zncc,
                "max_shift_px": refine_defaults.max_shift_px,
                "max_keypoint_uncertainty": refine_defaults.max_keypoint_uncertainty,
            }),
        }),
        two_view_geometries: None,
    }
}
