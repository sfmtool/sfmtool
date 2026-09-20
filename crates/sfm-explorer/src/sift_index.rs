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
//! the capture and is a background task the person asks for.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use sfmtool_core::features::kdforest::{
    KdForestParams, KdForestU8, KdfError, KdfSiftSources, KdfWorkspaceContents,
    KdfWorkspaceMetadata, KdfWriteOptions, LazyKdForestOptions, LazyKdForestU8,
};
use sfmtool_core::progress::Progress;
use sfmtool_core::{progress_note, SfmrReconstruction};

use crate::action_log::{Actor, Kind};
use crate::background::{Finished, Job, Operation};
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

/// Which of the three states a node's index is in.
///
/// `none` is a node with no file at its index path and none opened by hand;
/// `current` is an index over exactly this node's images, in this node's order,
/// from the `.sift` files that are on disk now; anything else is `stale`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SiftIndexState {
    None,
    Current,
    Stale,
}

impl SiftIndexState {
    /// The word the row and the wire both use.
    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Current => "current",
            Self::Stale => "stale",
        }
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

/// `path` as the index of `node` may be written to, or why it may not.
///
/// A caller naming the file is worth keeping -- a second index over the same
/// capture, under a name of its own, is a reasonable thing for an agent to ask
/// for -- and what it is not worth is writing anywhere on the disk. So the path
/// is resolved against the directory holding the node's `.sfmr` when it is
/// relative, lexically normalised (`.` and `..` folded, no filesystem walked,
/// since the file is not there yet), and refused when the answer leaves that
/// directory.
///
/// Lexical rather than canonical on purpose: `canonicalize` needs the path to
/// exist, and this one is about to be created. A symlink out of the directory
/// therefore passes, which is a thing the person who made the symlink asked
/// for.
fn within_index_dir(node: &SceneNode, path: &Path) -> Result<PathBuf, String> {
    let home = index_path(node)
        .and_then(|index| index.parent().map(Path::to_path_buf))
        .ok_or_else(|| unsaved_refusal(node))?;
    let joined = if path.is_absolute() {
        path.to_path_buf()
    } else {
        home.join(path)
    };
    let resolved = lexically_normalized(&joined);
    if resolved.starts_with(&home) {
        return Ok(resolved);
    }
    Err(format!(
        "{} is outside {}, and the SIFT index is written beside the .sfmr file.",
        resolved.display(),
        home.display()
    ))
}

/// What a node with no path on disk is told when it is asked for an index.
pub(crate) fn unsaved_refusal(node: &SceneNode) -> String {
    format!(
        "Save {} first: the SIFT index is written beside the .sfmr file.",
        node.label
    )
}

/// `path` with its `.` and `..` components folded and its separators this
/// platform's, without touching the filesystem.
///
/// A `..` that would climb past the root is dropped, so the answer never
/// escapes upwards through a prefix.
fn lexically_normalized(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for component in path.components() {
        match component {
            std::path::Component::CurDir => {}
            std::path::Component::ParentDir => {
                if !out.pop() {
                    // Nothing left to climb: keep the `..` so a relative path
                    // that really does leave its root is not silently flattened
                    // into one that does not.
                    out.push("..");
                }
            }
            other => out.push(other.as_os_str()),
        }
    }
    out
}

/// The `.sift` file of every image of `recon` that has one, by image index.
fn readable_sift_files(recon: &SfmrReconstruction) -> Vec<(u32, PathBuf)> {
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
    pub(crate) fn sift_index_state(&self, id: ReconId) -> SiftIndexState {
        match self.sift_index(id) {
            None => SiftIndexState::None,
            Some(index) if index.stale.is_some() => SiftIndexState::Stale,
            Some(_) => SiftIndexState::Current,
        }
    }

    /// Where `id`'s index goes when nobody names a path.
    pub(crate) fn sift_index_path(&self, id: ReconId) -> Option<PathBuf> {
        index_path(self.node(id)?)
    }

    /// Whether the operation running on `id` is the index build.
    pub(crate) fn building_sift_index(&self, id: ReconId) -> bool {
        self.background_task().is_some_and(|task| {
            task.node == id && task.operation.name == Operation::BUILD_SIFT_INDEX.name
        })
    }

    /// Look for `id`'s index if nothing has yet, and re-derive its state when
    /// the node's image table has moved since the state was derived.
    ///
    /// What the Track Edit panel, the Scene tree and the first item put on a
    /// node's bench call. A `.kdf` opens without decoding a tree or a
    /// descriptor block, so looking costs a stat and a header read, and a
    /// session that finds the file the last one built is a session that can
    /// search. It **builds nothing** and says nothing when there is no file: an
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
        let opened = self.open_sift_index(id, Some(path));
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
    }

    /// Open `path`, or the node's own index path when none is named, as `id`'s
    /// SIFT index.
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
        path: Option<PathBuf>,
    ) -> Result<PathBuf, String> {
        let node = self
            .node(id)
            .ok_or_else(|| "That reconstruction is no longer loaded.".to_string())?;
        let path = match path {
            Some(path) => path,
            None => index_path(node).ok_or_else(|| unsaved_refusal(node))?,
        };
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
        Ok(path)
    }

    /// Let go of `id`'s index, leaving the file where it is.
    ///
    /// Recorded rather than silent: the search entry greys the moment this
    /// happens, and a person reading the log should find out why.
    pub(crate) fn close_sift_index(&mut self, id: ReconId) -> Result<(), String> {
        let node = self
            .node(id)
            .ok_or_else(|| "That reconstruction is no longer loaded.".to_string())?;
        let label = node.label.clone();
        let Some(index) = self.sift_indexes.get(&id).and_then(Option::as_ref) else {
            return Err(format!("No SIFT index is open beside {label}."));
        };
        let path = index.path.clone();
        // `Some(None)`, not removed: the miss is what says nothing should stat
        // that path again this session, and a person who just closed an index
        // does not want it re-opened by the next frame that draws the row.
        self.sift_indexes.insert(id, None);
        self.action_log.record(
            Kind::Bench,
            format!("Closed the SIFT index of {label}: {}", path.display()),
        );
        Ok(())
    }

    /// Why `id` cannot have an index built for it, or `None` when it can.
    ///
    /// Asked by the step itself, so the menu entry and the task cannot disagree
    /// about when there is something to index.
    pub(crate) fn build_sift_index_refusal(&self, id: ReconId) -> Option<String> {
        self.busy_refusal(id)
            .or_else(|| self.sift_index_home_refusal(id))
            .or_else(|| self.sift_sources_refusal(id))
    }

    /// Why there is nowhere to put `id`'s index, or `None` when there is.
    pub(crate) fn sift_index_home_refusal(&self, id: ReconId) -> Option<String> {
        let node = self.node(id)?;
        node.path.is_none().then(|| unsaved_refusal(node))
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
                "No SIFT index is open. Build one from the SIFT Index row in the Scene \
                 tree, or from this menu."
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

    /// Build a SIFT index over every `.sift` file of `id`, write it beside the
    /// node's `.sfmr`, and open it, on a worker thread.
    ///
    /// The corpus carries **one image-table row per image of the node**, in the
    /// node's own order, including the images that have no `.sift` file: an
    /// image with no features contributes no descriptor and still takes its
    /// row, which is what keeps a corpus image index and a node image index the
    /// same number.
    ///
    /// `progress` names three phases: `read descriptors`, `build forest` and
    /// `write index`, each reporting within its own share.
    pub(crate) fn start_build_sift_index(
        &mut self,
        id: ReconId,
        path: Option<PathBuf>,
    ) -> Result<(), String> {
        let outcome = self.begin_build_sift_index(id, path);
        if let Err(message) = &outcome {
            self.action_log.fail(Kind::Bench, message.clone());
        }
        outcome
    }

    fn begin_build_sift_index(&mut self, id: ReconId, path: Option<PathBuf>) -> Result<(), String> {
        let job = self.build_sift_index_job(id, path)?;
        self.start_background_task(Operation::BUILD_SIFT_INDEX, id, job)
    }

    /// The work a build does, as a closure that owns everything it reads.
    ///
    /// Separated from starting it for the reason every other operation's job is:
    /// the refusals and the plan are the state's to work out, and what crosses
    /// to the worker is one closure holding no reference into the scene.
    pub(crate) fn build_sift_index_job(
        &self,
        id: ReconId,
        path: Option<PathBuf>,
    ) -> Result<Job, String> {
        if let Some(why) = self.build_sift_index_refusal(id) {
            return Err(why);
        }
        let node = self
            .node(id)
            .ok_or_else(|| "That reconstruction is no longer loaded.".to_string())?;
        let recon = node.recon();
        let path = match path {
            Some(path) => within_index_dir(node, &path)?,
            None => index_path(node).ok_or_else(|| unsaved_refusal(node))?,
        };
        let plan = BuildPlan {
            path,
            sources: readable_sift_files(recon),
            image_names: recon
                .image_table
                .images
                .iter()
                .map(|image| image.name.clone())
                .collect(),
            workspace: workspace_metadata(recon),
        };
        Ok(Box::new(move |progress| build(plan, progress)))
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
    }

    /// Forget the index of a node that has left the scene.
    pub(crate) fn forget_sift_index(&mut self, id: ReconId) {
        self.sift_indexes.remove(&id);
    }
}

/// Everything the build needs, owned, so the worker holds no reference into the
/// scene.
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

/// The forest build itself: read, build, write, reopen.
fn build(plan: BuildPlan, progress: &Progress<'_>) -> Finished {
    let BuildPlan {
        path,
        sources,
        image_names,
        workspace,
    } = plan;

    let images = image_names.len();
    let mut descriptors: Vec<u8> = Vec::new();
    let mut origins = Vec::new();
    let mut geometry = Vec::new();
    // Zeros for an image with no `.sift` file, which is what a later staleness
    // check expects to find for one.
    let mut feature_tool_hashes = vec![[0u8; 16]; images];
    let mut sift_content_hashes = vec![[0u8; 16]; images];
    let mut dimension = 0usize;
    // How the time divides on a 370-image, 3M-descriptor capture: about 1.7 s,
    // 3.4 s and 4.5 s, or 0.18 / 0.36 / 0.46. On a 4054-image, 40M-descriptor
    // one the forest grows fastest of the three and lands nearer 0.24 / 0.47 /
    // 0.29, so these shares sit between the two shapes rather than fitting
    // either exactly. Each stage reports within its own share -- the read per
    // image, the forest per leaf, the write per batch of blocks -- so the bar
    // moves through all three.
    let [read, forest_share, write] = progress.split([0.2, 0.4, 0.4]);
    {
        let mut phase = read.phase("read descriptors");
        for (index, (image, sift_path)) in sources.iter().enumerate() {
            if phase.is_cancelled() {
                return Finished::Cancelled;
            }
            phase.set_fraction(index as f32 / sources.len() as f32);
            let data = match sfmtool_sift_format::read_sift_partial(sift_path, usize::MAX) {
                Ok(data) => data,
                Err(e) => {
                    return Finished::Failed(format!("Cannot read {}: {e}", sift_path.display()))
                }
            };
            // The identities the file records, read off the very archive the
            // descriptors came out of: the staleness test compares these
            // against the `.sift` on disk, so an index built here and never
            // touched since reads as current.
            if let Some(hash) = decode_xxh128(&data.content_hash.content_xxh128) {
                sift_content_hashes[*image as usize] = hash;
            }
            if let Some(hash) = decode_xxh128(&data.content_hash.feature_tool_xxh128) {
                feature_tool_hashes[*image as usize] = hash;
            }
            let rows = data.descriptors.nrows();
            let dim = data.descriptors.ncols();
            if rows == 0 {
                continue;
            }
            if dimension == 0 {
                dimension = dim;
            } else if dim != dimension {
                return Finished::Failed(format!(
                    "{} holds {dim}-byte descriptors and the rest hold {dimension}-byte ones.",
                    sift_path.display()
                ));
            }
            for row in 0..rows {
                descriptors.extend(data.descriptors.row(row).iter().copied());
                origins.push(sfmtool_core::features::kdforest::FeatureOrigin {
                    image_index: *image,
                    image_feature_index: row as u32,
                });
                geometry.push([
                    [data.positions_xy[[row, 0]], data.positions_xy[[row, 1]]],
                    [
                        data.affine_shapes[[row, 0, 0]],
                        data.affine_shapes[[row, 0, 1]],
                    ],
                    [
                        data.affine_shapes[[row, 1, 0]],
                        data.affine_shapes[[row, 1, 1]],
                    ],
                ]);
            }
        }
        progress_note!(phase, "{} descriptors", origins.len());
    }
    if origins.is_empty() {
        return Finished::Failed(
            "Every .sift file of this reconstruction is empty, so there is nothing to index."
                .to_string(),
        );
    }

    let count = origins.len();
    let forest = {
        let phase = forest_share.phase("build forest");
        match KdForestU8::build(
            &descriptors,
            count,
            dimension,
            KdForestParams::default(),
            &phase,
        ) {
            Ok(forest) => forest,
            Err(_) => return Finished::Cancelled,
        }
    };

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
                return Finished::Failed(format!("Cannot create {}: {e}", parent.display()));
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
            return match e {
                KdfError::Cancelled(_) => Finished::Cancelled,
                e => Finished::Failed(format!("Cannot write {}: {e}", path.display())),
            };
        }
    }

    let opened = match LazyKdForestU8::open(&path, LazyKdForestOptions::default()) {
        Ok(forest) => forest,
        Err(e) => {
            return Finished::Failed(format!(
                "Wrote {} and then could not open it: {e}",
                path.display()
            ))
        }
    };
    Finished::SiftIndex {
        text: format!(
            "Built the SIFT index {}: {count} descriptors of {images} images",
            path.display()
        ),
        path,
        forest: Arc::new(opened),
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
fn image_fingerprint(recon: &SfmrReconstruction) -> u64 {
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
fn decode_xxh128(digest: &str) -> Option<[u8; 16]> {
    (digest.len() == 32)
        .then(|| u128::from_str_radix(digest, 16).ok())
        .flatten()
        .map(u128::to_be_bytes)
}
