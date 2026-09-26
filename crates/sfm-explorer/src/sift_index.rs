// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The SIFT index of a reconstruction: the `.kdf` a bench search queries, where
//! it lives, how one is opened, how one is built, and what says it is still
//! good.
//!
//! See `specs/gui/sift-index.md`. The index is a **node's**, not a panel's: a
//! bench search is a step on that node's bench and every track on it queries
//! the same forest, so what is held here is one open forest per loaded node,
//! beside the node's own `.sfmr` file.
//!
//! The viewer never builds one behind the person's back. Opening is free --
//! a `.kdf` is opened without decoding a tree or a descriptor block
//! (`specs/core/features/lazy-kdforest-query.md`) -- so the node's index path is
//! opened on sight when the file is there; building one reads every `.sift` of
//! the capture and is the first half of the background task that builds the
//! node's index files ([`crate::index_files`]).

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use rayon::prelude::*;
use sfmtool_core::features::kdforest::{
    FeatureGeometry, FeatureOrigin, KdForestParams, KdForestU8, KdfError, KdfSiftSources,
    KdfWorkspaceContents, KdfWorkspaceMetadata, KdfWriteOptions, LazyKdForestOptions,
    LazyKdForestU8,
};
use sfmtool_core::progress::Progress;
use sfmtool_core::{progress_note, SfmrReconstruction};
use sfmtool_sift_format::DESCRIPTOR_DIM;

use crate::action_log::{Actor, Kind};
use crate::index_files::{unsaved_refusal, IndexFileState, Stopped};
use crate::scene::{ReconId, SceneNode};
use crate::state::AppState;

#[cfg(test)]
pub(crate) mod tests;

/// What a reconstruction's index file is called, after the `.sfmr`'s own stem.
pub(crate) const INDEX_FILE_SUFFIX: &str = "-sift-index.kdf";

/// One node's open SIFT index, and what the node makes of it.
#[derive(Clone)]
pub(crate) struct SiftIndex {
    /// The `.kdf` it was opened from, as the tree and the wire name it.
    pub(crate) path: PathBuf,
    /// The forest itself.
    ///
    /// Behind an [`Arc`] because a search runs on a **worker**: the forest owns
    /// a query thread pool and a bounded decoded-block cache, so a task takes a
    /// clone of the handle rather than opening the file a second time and
    /// paying for a second cache.
    pub(crate) forest: Arc<LazyKdForestU8>,
    /// How many images the corpus carries, which is the node's own count when
    /// the index is current.
    pub(crate) images: usize,
    /// Why this index is not an index of the node as it stands, or `None` when
    /// it is. Derived when the file is opened or built, and again when a
    /// version lands whose image table differs -- never per frame.
    stale: Option<String>,
    /// Fingerprint of the node's image table the state above was derived
    /// against, which is what says a version has moved the question.
    node_images: u64,
    /// The version the state above was derived against, so a frame that pushed
    /// nothing costs a `u64` comparison.
    checked_at: u64,
}

impl SiftIndex {
    /// How many descriptors the corpus holds.
    pub(crate) fn feature_count(&self) -> usize {
        self.forest.len()
    }

    /// Why this index will not do for the node it is open beside, or `None`.
    pub(crate) fn stale_reason(&self) -> Option<&str> {
        self.stale.as_deref()
    }
}

/// Where `node`'s index goes: the `.sfmr`'s sibling, named after its stem.
///
/// Beside the reconstruction rather than beside the features, because the
/// forest is a fact about a reconstruction: two reconstructions saved in one
/// directory have two indexes, and a reconstruction whose images span several
/// image directories has one index in one obvious place.
///
/// `None` for a node with no path on disk -- the demo scene, a value never
/// saved -- which has nowhere to put one.
///
/// **The separators are this platform's, all of them.** A `.sfmr` path the
/// session was handed can be spelled either way, and a path joined onto it
/// reads in a hover text, a reply and a log row as two conventions arguing.
/// Rebuilding the path from its components is what settles it:
/// `Path::components` splits on both separators and `PathBuf` re-joins with the
/// platform's own.
pub(crate) fn index_path(node: &SceneNode) -> Option<PathBuf> {
    let path = node.path.as_ref()?;
    let stem = path.file_stem()?;
    let mut name = stem.to_os_string();
    name.push(INDEX_FILE_SUFFIX);
    Some(normalized(&path.with_file_name(name)))
}

/// `path` spelled with this platform's separator throughout.
fn normalized(path: &Path) -> PathBuf {
    path.components().collect()
}

/// The `.sift` file of every image of `recon` that has one, by image index.
pub(crate) fn readable_sift_files(recon: &SfmrReconstruction) -> Vec<(u32, PathBuf)> {
    (0..recon.image_table.images.len())
        .map(|image| (image as u32, recon.sift_path_for_image(image)))
        .filter(|(_, path)| path.is_file())
        .collect()
}

impl AppState {
    /// The SIFT index open beside `id`, whatever state it is in, or `None` when
    /// none is.
    pub(crate) fn sift_index(&self, id: ReconId) -> Option<&SiftIndex> {
        self.sift_indexes.get(&id)?.as_ref()
    }

    /// Which of the three states `id`'s index is in.
    pub(crate) fn sift_index_state(&self, id: ReconId) -> IndexFileState {
        match self.sift_index(id) {
            None => IndexFileState::None,
            Some(index) if index.stale.is_some() => IndexFileState::Stale,
            Some(_) => IndexFileState::Current,
        }
    }

    /// Where `id`'s index goes when nobody names a path.
    pub(crate) fn sift_index_path(&self, id: ReconId) -> Option<PathBuf> {
        index_path(self.node(id)?)
    }

    /// Look for `id`'s index if nothing has yet, and re-derive its state when
    /// the node's image table has moved since the state was derived.
    ///
    /// Called through [`Self::refresh_index_files`], which is what Track View,
    /// the Scene tree and the first item put on a node's bench call, and which
    /// asks the same of the cluster-patches file after it. A `.kdf` opens
    /// without decoding a tree or a descriptor block, so looking costs a stat
    /// and a header read, and a session that finds the file the last one built
    /// is a session that can search. It **builds nothing** and says nothing when there is no file: an
    /// index that is not there is the normal state of a workspace, not a
    /// refusal.
    pub(crate) fn refresh_sift_index(&mut self, id: ReconId) {
        if self.sift_indexes.contains_key(&id) {
            self.recheck_sift_index(id);
            return;
        }
        let path = self.sift_index_path(id).filter(|p| p.is_file());
        let Some(path) = path else {
            // Remembered as a miss, so the panel's every frame is not a stat of
            // a file the reconstruction does not have beside it.
            self.sift_indexes.insert(id, None);
            return;
        };
        // **The row this writes is the viewer's.** Nobody asked for it: the look
        // happens because something was put on the bench or because the tree
        // drew a row, in the middle of the step that did it. Attributed to
        // whoever was acting, it would be the last row that step wrote, and a
        // wire reply that reports the step's own sentence would report this one
        // instead.
        let standing = self.action_log.actor();
        self.action_log.set_actor(Actor::Viewer);
        let opened = self.open_sift_index(id, path);
        if let Err(why) = opened {
            // Said out loud: a file sitting where the index goes and not opening
            // at all is worth a row, and this is the one chance to write it --
            // the miss is remembered, so nothing looks again.
            self.sift_indexes.insert(id, None);
            self.action_log.fail(Kind::Bench, why);
        }
        self.action_log.set_actor(standing);
    }

    /// Re-derive the state of an open index when the node's image table has
    /// moved under it.
    ///
    /// The version serial is the cheap question and the image names are the
    /// real one: a version that edits geometry leaves the corpus's claim about
    /// the node exactly as true as it was, and only one that adds, removes or
    /// renames an image can change the answer. The `.sift` files on disk are
    /// **not** re-read outside this; a rebuild and a re-open are the two ways to
    /// ask about them again.
    fn recheck_sift_index(&mut self, id: ReconId) {
        let Some(index) = self.sift_indexes.get(&id).and_then(Option::as_ref) else {
            return;
        };
        let Some(node) = self.node(id) else {
            return;
        };
        let serial = node.history.current_version().serial.as_u64();
        if index.checked_at == serial {
            return;
        }
        let fingerprint = image_fingerprint(node.recon());
        let moved = fingerprint != index.node_images;
        // Re-derived only when the table the judgement was about has moved,
        // which is the read of every `.sift` metadata; otherwise the version is
        // noted and the verdict stands.
        let (stale, images) = match moved {
            false => (None, 0),
            true => (
                staleness(&index.forest, node.recon(), &index.path),
                corpus_images(&index.forest),
            ),
        };
        let index = self
            .sift_indexes
            .get_mut(&id)
            .and_then(Option::as_mut)
            .expect("just read");
        if moved {
            index.stale = stale;
            index.images = images;
            index.node_images = fingerprint;
        }
        index.checked_at = serial;
        if moved {
            self.recheck_cluster_patches(id);
        }
    }

    /// Open `path` as `id`'s SIFT index.
    ///
    /// **A file that opens is adopted**, current or stale: a person who asked
    /// for this one wants to see what is in it and why it will not do, and the
    /// row says which. What a stale index does not do is answer a search --
    /// [`Self::sift_index_search_refusal`] is the one gate on that, so a match's
    /// corpus image index and the candidate's node image index cannot come
    /// apart.
    pub(crate) fn open_sift_index(
        &mut self,
        id: ReconId,
        path: PathBuf,
    ) -> Result<PathBuf, String> {
        let node = self
            .node(id)
            .ok_or_else(|| crate::state::NOT_LOADED.to_string())?;
        let forest = LazyKdForestU8::open(&path, LazyKdForestOptions::default())
            .map_err(|e| format!("Cannot open {}: {e}", path.display()))?;
        let stale = staleness(&forest, node.recon(), &path);
        let label = node.label.clone();
        let serial = node.history.current_version().serial.as_u64();
        let fingerprint = image_fingerprint(node.recon());
        let features = forest.len();
        let images = corpus_images(&forest);
        let text = match &stale {
            None => format!(
                "Opened the SIFT index of {label}: {} with {features} descriptors",
                path.display()
            ),
            Some(why) => format!(
                "Opened the SIFT index of {label}: {} -- it is out of date. {why}",
                path.display()
            ),
        };
        self.sift_indexes.insert(
            id,
            Some(SiftIndex {
                path: path.clone(),
                forest: Arc::new(forest),
                images,
                stale,
                node_images: fingerprint,
                checked_at: serial,
            }),
        );
        self.action_log.record(Kind::Bench, text);
        // The cluster-patches file is judged against the index that is open,
        // so a different one open is a different answer for it too.
        self.recheck_cluster_patches(id);
        Ok(path)
    }

    /// Why there is nothing to index beside `id`, or `None` when there is.
    ///
    /// **Stats `.sift` paths** until it finds one, so a capture with features
    /// costs a single stat and one without costs one per image. Cheap enough to
    /// ask from a menu that is open, which is where the greying happens.
    pub(crate) fn sift_sources_refusal(&self, id: ReconId) -> Option<String> {
        let node = self.node(id)?;
        let recon = node.recon();
        let any = (0..recon.image_table.images.len())
            .any(|image| recon.sift_path_for_image(image).is_file());
        (!any).then(|| {
            format!(
                "No .sift file of {} could be found, so there are no descriptors to index.",
                node.label
            )
        })
    }

    /// Why a search cannot query `id`'s index, or `None` when it can.
    ///
    /// The one place the three states turn into a yes or a no for the search,
    /// so the greyed entry, the step and the wire all say it the same way.
    pub(crate) fn sift_index_search_refusal(&self, id: ReconId) -> Option<String> {
        match self.sift_index(id) {
            None => Some(
                "No SIFT index is open. Build the index files from the Index Files row in \
                 the Scene tree, or from this menu."
                    .to_string(),
            ),
            Some(index) => index.stale.as_ref().map(|why| {
                format!(
                    "The SIFT index {} is out of date. {why}",
                    index.path.display()
                )
            }),
        }
    }

    /// Install an index a background build produced.
    ///
    /// Classified by the same function an opened file goes through, rather than
    /// taken as current because the build just made it: one answer to "is this
    /// an index of this node" means a build that raced an edit cannot leave the
    /// row claiming something the search would not honour.
    pub(crate) fn install_sift_index(
        &mut self,
        id: ReconId,
        path: PathBuf,
        forest: Arc<LazyKdForestU8>,
    ) {
        let Some(node) = self.node(id) else {
            return;
        };
        let stale = staleness(&forest, node.recon(), &path);
        let checked_at = node.history.current_version().serial.as_u64();
        let node_images = image_fingerprint(node.recon());
        let images = corpus_images(&forest);
        self.sift_indexes.insert(
            id,
            Some(SiftIndex {
                path,
                forest,
                images,
                stale,
                node_images,
                checked_at,
            }),
        );
        self.recheck_cluster_patches(id);
    }

    /// Forget the index of a node that has left the scene.
    pub(crate) fn forget_sift_index(&mut self, id: ReconId) {
        self.sift_indexes.remove(&id);
    }
}

/// What the index half of the index-files build writes, and from what.
///
/// The corpus carries **one image-table row per image of the node**, in the
/// node's own order, including the images that have no `.sift` file: an image
/// with no features contributes no descriptor and still takes its row, which
/// is what keeps a corpus image index and a node image index the same number.
///
/// Everything owned, so the worker holds no reference into the scene.
pub(crate) struct BuildPlan {
    /// Where the `.kdf` goes.
    path: PathBuf,
    /// The images that have a `.sift` file, by node image index.
    sources: Vec<(u32, PathBuf)>,
    /// Every image of the node, in its own order, so a corpus image index is a
    /// node image index.
    image_names: Vec<String>,
    /// The workspace the features were extracted under, which the file records.
    workspace: KdfWorkspaceMetadata,
}

impl BuildPlan {
    /// The plan for `node`'s index, written at the node's own index path.
    pub(crate) fn of(node: &SceneNode) -> Result<Self, String> {
        let recon = node.recon();
        let path = index_path(node).ok_or_else(|| unsaved_refusal(node))?;
        Ok(Self {
            path,
            sources: readable_sift_files(recon),
            image_names: recon
                .image_table
                .images
                .iter()
                .map(|image| image.name.clone())
                .collect(),
            workspace: workspace_metadata(recon),
        })
    }
}

/// An index the build wrote and opened.
pub(crate) struct BuiltIndex {
    /// The `.kdf` that was written.
    pub(crate) path: PathBuf,
    /// It, opened the way a lazy open opens one.
    pub(crate) forest: LazyKdForestU8,
    /// How many descriptors it holds.
    pub(crate) descriptors: usize,
    /// How many images its table has a row for.
    pub(crate) images: usize,
}

/// The forest build itself: read, build, write, reopen.
///
/// `progress` names three phases: `read descriptors`, `build forest` and
/// `write index`, each reporting within its own share.
pub(crate) fn build_index(plan: BuildPlan, progress: &Progress<'_>) -> Result<BuiltIndex, Stopped> {
    let BuildPlan {
        path,
        sources,
        image_names,
        workspace,
    } = plan;

    let images = image_names.len();
    // How the time divides on a 370-image, 3M-descriptor capture: about 0.2 s
    // of reading against 3.4 s of forest and some 2 s of writing, or
    // 0.04 / 0.62 / 0.35. On a 4054-image, 40M-descriptor one the read stays
    // the smallest of the three at 4.4 s against 46 s and 30 s, or
    // 0.05 / 0.57 / 0.37, so the two shapes agree to within a sixteenth and
    // these shares are the sixteenths between them. Exact
    // sixteenths, because the weights are normalised in `f32` and three that
    // sum to one only approximately leave the finished bar a hair short of its
    // end. Each stage reports within its own share -- the read per image, the
    // forest per leaf, the write per batch of blocks -- so the bar moves
    // through all three.
    let [read, forest_share, write] = progress.split([0.0625, 0.5625, 0.375]);
    let Corpus {
        descriptors,
        origins,
        geometry,
        feature_tool_hashes,
        sift_content_hashes,
    } = read_corpus(&sources, images, &read)?;
    if origins.is_empty() {
        return Err(Stopped::Failed(
            "Every .sift file of this reconstruction is empty, so there is nothing to index."
                .to_string(),
        ));
    }

    let count = origins.len();
    let forest = {
        let phase = forest_share.phase("build forest");
        match KdForestU8::build(
            &descriptors,
            count,
            DESCRIPTOR_DIM,
            KdForestParams::default(),
            &phase,
        ) {
            Ok(forest) => forest,
            Err(_) => return Err(Stopped::Cancelled),
        }
    };
    // The forest keeps its own copy of the descriptors, so from here the corpus
    // buffer is a second one nothing reads. On a 40M-descriptor capture that is
    // 5.2 GB, and the write that follows is the longest stage of the build.
    drop(descriptors);

    let sources = KdfSiftSources {
        workspace,
        image_names,
        feature_tool_hashes,
        sift_content_hashes,
        origins,
        geometry,
    };
    {
        let phase = write.phase("write index");
        if let Some(parent) = path.parent() {
            if let Err(e) = std::fs::create_dir_all(parent) {
                return Err(Stopped::Failed(format!(
                    "Cannot create {}: {e}",
                    parent.display()
                )));
            }
        }
        // Straight at the index path, replacing what is there: a rebuild is the
        // person asking for this index rather than for a second one. The writer
        // streams into a temporary sibling and renames over the target at the
        // end, so the index that is open stays exactly as it was until that
        // instant -- a rebuild cancelled ten seconds in, or failing on its last
        // block, leaves the working index standing and nothing beside it.
        let options = KdfWriteOptions {
            replace_existing: true,
            ..KdfWriteOptions::default()
        };
        let written = forest.write_kdf(&path, Some(&sources), &options, &phase);
        if let Err(e) = written {
            return Err(match e {
                KdfError::Cancelled(_) => Stopped::Cancelled,
                e => Stopped::Failed(format!("Cannot write {}: {e}", path.display())),
            });
        }
    }

    let opened = LazyKdForestU8::open(&path, LazyKdForestOptions::default()).map_err(|e| {
        Stopped::Failed(format!(
            "Wrote {} and then could not open it: {e}",
            path.display()
        ))
    })?;
    Ok(BuiltIndex {
        path,
        forest: opened,
        descriptors: count,
        images,
    })
}

/// One capture's `.sift` files, read into the arrays the forest and the file
/// are made of.
struct Corpus {
    /// Every descriptor of every image end to end, [`DESCRIPTOR_DIM`] bytes to
    /// the row, in image order and within an image in feature order.
    descriptors: Vec<u8>,
    /// Which image and which of its features each corpus row came from.
    origins: Vec<FeatureOrigin>,
    /// Each corpus row's keypoint centre and affine shape.
    geometry: Vec<FeatureGeometry>,
    /// Per **node** image, zero for one with no `.sift` file.
    feature_tool_hashes: Vec<[u8; 16]>,
    /// Per **node** image, zero for one with no `.sift` file, which is what a
    /// later staleness check expects to find for one.
    sift_content_hashes: Vec<[u8; 16]>,
}

/// The two identities one `.sift` file records, as the image table holds them.
struct Identities {
    feature_tool: Option<[u8; 16]>,
    content: Option<[u8; 16]>,
}

/// Where one image's features go, handed to whichever worker reads that image.
///
/// The two slices are that image's own stretch of the corpus, disjoint from
/// every other slot's, so the files are read in whatever order the workers take
/// them and every row still lands where image order puts it.
struct Slot<'a> {
    path: &'a Path,
    /// `features * DESCRIPTOR_DIM` bytes.
    descriptors: &'a mut [u8],
    /// `features` rows, which is what says how many the file is expected to
    /// hold.
    geometry: &'a mut [FeatureGeometry],
}

/// What one image's read came back with: the value, `None` for a file the read
/// skipped because the build is stopping, and `Err` with what to say about it.
type Outcome<T> = Result<Option<T>, String>;

/// Read every `.sift` file of the capture into one corpus, in image order.
///
/// The files are read in parallel -- each is an open, a seek and a zstd expand,
/// with nothing shared between two of them -- and the order they come back in
/// is not the order they are placed. **Two passes are what keeps that from
/// costing a second copy of the corpus.** The first reads each file's metadata,
/// which is a few hundred bytes and says how many features the file holds; the
/// counts give every image its offset, so the second pass expands straight into
/// its own stretch of one buffer sized exactly once. Collecting whole images
/// and concatenating them afterwards would hold the capture twice over, and on
/// a 40M-descriptor capture one copy is 5.2 GB.
///
/// The two passes are the phases `count features` and `read features`, each
/// reporting a count per image through a counter its workers share. A cancel is
/// read before each file, so a stopping build finishes at most one file per
/// worker and hands back nothing. When several files fail, the one named is the
/// first in image order, whichever worker reached it first.
fn read_corpus(
    sources: &[(u32, PathBuf)],
    images: usize,
    progress: &Progress<'_>,
) -> Result<Corpus, Stopped> {
    let mut phase = progress.phase("read descriptors");
    // Counting is a twelfth of the read on a warm cache and rather more on a
    // cold one, where it is the pass that pays for finding 4054 files at all:
    // 0.38 s against 3.97 s warm, 1.53 s against 4.11 s cold.
    let [counting, reading] = phase.split([0.125, 0.875]);
    let counts = feature_counts(sources, &counting)?;
    let corpus = read_features(sources, &counts, images, &reading)?;
    progress_note!(phase, "{} descriptors", corpus.origins.len());
    Ok(corpus)
}

/// How many features each source holds, off its metadata alone.
fn feature_counts(
    sources: &[(u32, PathBuf)],
    progress: &Progress<'_>,
) -> Result<Vec<usize>, Stopped> {
    let phase = progress.phase("count features");
    let tally = Tally::new(&phase, sources.len());
    let counted: Vec<Outcome<usize>> = sources
        .par_iter()
        .map(|(_, path)| {
            if phase.is_cancelled() {
                return Ok(None);
            }
            let count = match sfmtool_sift_format::read_sift_metadata(path) {
                Ok((_, metadata, _)) => metadata.feature_count as usize,
                Err(e) => return Err(format!("Cannot read {}: {e}", path.display())),
            };
            tally.one();
            Ok(Some(count))
        })
        .collect();
    settle(counted, &phase)
}

/// Expand every file's features into one corpus, `counts` having said where
/// each image's rows go.
fn read_features(
    sources: &[(u32, PathBuf)],
    counts: &[usize],
    images: usize,
    progress: &Progress<'_>,
) -> Result<Corpus, Stopped> {
    let phase = progress.phase("read features");
    let rows: usize = counts.iter().sum();
    let mut descriptors = vec![0u8; rows * DESCRIPTOR_DIM];
    let mut geometry = vec![[[0.0f32; 2]; 3]; rows];
    // Which image and which feature a row is depends on the counts alone, so
    // the column is filled here rather than by the workers.
    let mut origins = Vec::with_capacity(rows);
    for ((image, _), count) in sources.iter().zip(counts) {
        origins.extend((0..*count as u32).map(|feature| FeatureOrigin {
            image_index: *image,
            image_feature_index: feature,
        }));
    }

    let tally = Tally::new(&phase, sources.len());
    let mut descriptor_rest = descriptors.as_mut_slice();
    let mut geometry_rest = geometry.as_mut_slice();
    let mut slots = Vec::with_capacity(sources.len());
    for ((_, path), count) in sources.iter().zip(counts) {
        let (mine, rest) = descriptor_rest.split_at_mut(count * DESCRIPTOR_DIM);
        descriptor_rest = rest;
        let (shapes, rest) = geometry_rest.split_at_mut(*count);
        geometry_rest = rest;
        slots.push(Slot {
            path,
            descriptors: mine,
            geometry: shapes,
        });
    }

    let read: Vec<Outcome<Identities>> = slots
        .into_par_iter()
        .map(|slot| {
            if phase.is_cancelled() {
                return Ok(None);
            }
            let data = match sfmtool_sift_format::read_sift_features(slot.path) {
                Ok(data) => data,
                Err(e) => return Err(format!("Cannot read {}: {e}", slot.path.display())),
            };
            // The file is opened twice, so a file rewritten between the two
            // reads is the one thing the offsets cannot absorb. It is caught
            // here rather than trusted, because the alternative is descriptors
            // landing in another image's rows.
            if data.positions_xy.len() != slot.geometry.len() {
                return Err(format!(
                    "{} holds {} features and held {} a moment ago, so it was written while \
                     this index was being built. Build it again.",
                    slot.path.display(),
                    data.positions_xy.len(),
                    slot.geometry.len()
                ));
            }
            slot.descriptors.copy_from_slice(&data.descriptors);
            for ((row, centre), shape) in slot
                .geometry
                .iter_mut()
                .zip(&data.positions_xy)
                .zip(&data.affine_shapes)
            {
                *row = [*centre, shape[0], shape[1]];
            }
            tally.one();
            // The identities the file records, read off the very archive the
            // descriptors came out of: the staleness test compares these
            // against the `.sift` on disk, so an index built here and never
            // touched since reads as current.
            Ok(Some(Identities {
                feature_tool: decode_xxh128(&data.content_hash.feature_tool_xxh128),
                content: decode_xxh128(&data.content_hash.content_xxh128),
            }))
        })
        .collect();
    let identities = settle(read, &phase)?;

    // Zeros for an image with no `.sift` file at all.
    let mut feature_tool_hashes = vec![[0u8; 16]; images];
    let mut sift_content_hashes = vec![[0u8; 16]; images];
    for ((image, _), identity) in sources.iter().zip(identities) {
        if let Some(hash) = identity.feature_tool {
            feature_tool_hashes[*image as usize] = hash;
        }
        if let Some(hash) = identity.content {
            sift_content_hashes[*image as usize] = hash;
        }
    }
    Ok(Corpus {
        descriptors,
        origins,
        geometry,
        feature_tool_hashes,
        sift_content_hashes,
    })
}

/// What a pass over the files came to, as the job's own answer.
///
/// A cancel outranks a failure: a build the person stopped hands back nothing
/// whatever else it ran into. Otherwise the failure reported is the first in
/// **image** order rather than the first a worker hit, so two unreadable files
/// name the same one on every run.
fn settle<T>(outcomes: Vec<Outcome<T>>, progress: &Progress<'_>) -> Result<Vec<T>, Stopped> {
    if progress.is_cancelled() {
        return Err(Stopped::Cancelled);
    }
    let mut values = Vec::with_capacity(outcomes.len());
    for outcome in outcomes {
        match outcome {
            Ok(Some(value)) => values.push(value),
            Ok(None) => return Err(Stopped::Cancelled),
            Err(message) => return Err(Stopped::Failed(message)),
        }
    }
    Ok(values)
}

/// The counter the workers of one pass report through.
///
/// They finish in whatever order the files come back in, so what moves the bar
/// is how many are done rather than which one a worker is on -- the same shape
/// the forest build's own counter has.
struct Tally<'p, 'a> {
    progress: &'p Progress<'a>,
    done: AtomicU64,
    total: u64,
}

impl<'p, 'a> Tally<'p, 'a> {
    fn new(progress: &'p Progress<'a>, total: usize) -> Self {
        Self {
            progress,
            done: AtomicU64::new(0),
            total: total as u64,
        }
    }

    /// Count one file, and say so.
    fn one(&self) {
        let done = self.done.fetch_add(1, Ordering::Relaxed) + 1;
        self.progress.count(done, Some(self.total), "image");
    }
}

/// The workspace the `.kdf` records, taken from the reconstruction's own.
fn workspace_metadata(recon: &SfmrReconstruction) -> KdfWorkspaceMetadata {
    let workspace = &recon.metadata.workspace;
    KdfWorkspaceMetadata {
        absolute_path: workspace.absolute_path.clone(),
        relative_path: workspace.relative_path.clone(),
        contents: KdfWorkspaceContents {
            feature_tool: workspace.contents.feature_tool.clone(),
            feature_type: workspace.contents.feature_type.clone(),
            feature_options: workspace.contents.feature_options.clone(),
            feature_prefix_dir: workspace.contents.feature_prefix_dir.clone(),
        },
    }
}

/// How many images the corpus carries, or `0` when it carries no table.
fn corpus_images(forest: &LazyKdForestU8) -> usize {
    forest
        .image_table()
        .ok()
        .flatten()
        .map_or(0, |table| table.names.len())
}

/// A digest of a reconstruction's image table, as the thing a staleness verdict
/// is about.
///
/// Touches no file, so a version that lands on a node costs this and nothing
/// else: only a table that has gained, lost, renamed or reordered an image can
/// change what the index's three tests answer, and that is exactly what this
/// value moves on.
pub(crate) fn image_fingerprint(recon: &SfmrReconstruction) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    recon.image_table.images.len().hash(&mut hasher);
    for image in &recon.image_table.images {
        image.name.hash(&mut hasher);
    }
    hasher.finish()
}

/// Why `forest` is not an index of `recon` as it stands, or `None` when it is.
///
/// **Every discrepancy is stale, an index over a superset of the node's images
/// included.** A forest query returns its nearest descriptors across the whole
/// corpus: hits from images the node no longer has would be dropped after the
/// fact, and they would already have taken the places of hits from images it
/// does have, so the answer differs from the one an index over the node's own
/// images gives.
///
/// The three tests are in cost order, and the sentence names the first that
/// fails. The third reads the content hash out of each `.sift` file's metadata
/// and decodes no feature; it compares against the file **on disk** rather than
/// against hashes held in the reconstruction, because an `embedded_patches`
/// reconstruction carries image hashes and no `.sift` hashes, and because the
/// search reads its query keypoints from the `.sift` on disk, so the file on
/// disk is the thing the index has to agree with.
fn staleness(forest: &LazyKdForestU8, recon: &SfmrReconstruction, path: &Path) -> Option<String> {
    let table = match forest.image_table() {
        Err(e) => {
            return Some(format!(
                "Cannot read the image table of {}: {e}",
                path.display()
            ))
        }
        Ok(None) => {
            return Some(format!(
                "{} carries no image table, so nothing says which photograph a hit came from.",
                path.display()
            ))
        }
        Ok(Some(table)) => table,
    };
    let images = &recon.image_table.images;
    if table.names.len() != images.len() {
        return Some(format!(
            "{} indexes {} images and this reconstruction has {}.",
            path.display(),
            table.names.len(),
            images.len()
        ));
    }
    for (index, (indexed, image)) in table.names.iter().zip(images.iter()).enumerate() {
        if indexed != &image.name {
            return Some(format!(
                "{} has {indexed:?} as image {index} and this reconstruction has {:?}; an \
                 index has to be over this reconstruction's own images, in its own order.",
                path.display(),
                image.name
            ));
        }
    }
    for (index, image) in images.iter().enumerate() {
        let recorded = table
            .sift_content_hashes
            .get(index)
            .copied()
            .unwrap_or([0u8; 16]);
        let sift = recon.sift_path_for_image(index);
        if !sift.is_file() {
            if recorded != [0u8; 16] {
                return Some(format!(
                    "{} has no .sift file now, and this index was built from one.",
                    image.name
                ));
            }
            continue;
        }
        let hash = match sfmtool_sift_format::read_sift_metadata(&sift) {
            Ok((_, _, hashes)) => decode_xxh128(&hashes.content_xxh128),
            Err(e) => {
                return Some(format!("Cannot read {}: {e}", sift.display()));
            }
        };
        let Some(hash) = hash else {
            return Some(format!(
                "{} records no readable content hash, so nothing says whether this index is \
                 over the features it holds now.",
                sift.display()
            ));
        };
        if recorded == [0u8; 16] {
            return Some(format!(
                "{} has a .sift file now, and this index was built without one.",
                image.name
            ));
        }
        if hash != recorded {
            return Some(format!(
                "The features of {} were extracted again after this index was built.",
                image.name
            ));
        }
    }
    None
}

/// A 32-character XXH128 hex digest as the 16 bytes a `.kdf` image table holds.
///
/// The first two hex digits are the first stored byte, which is the convention
/// every archive format here writes its hashes in. `None` for anything that is
/// not exactly 32 hex characters.
pub(crate) fn decode_xxh128(digest: &str) -> Option<[u8; 16]> {
    (digest.len() == 32)
        .then(|| u128::from_str_radix(digest, 16).ok())
        .flatten()
        .map(u128::to_be_bytes)
}
