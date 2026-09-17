// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The descriptor index beside a node: the `.kdf` a bench search queries, where
//! it lives, how one is opened, and how one is built.
//!
//! See `specs/gui/track-edit.md` § "The descriptor index" and
//! `specs/workspace/workspace.md` § "The Descriptor Index". The index is a
//! **node's**, not a panel's: a bench search is a step on that node's bench and
//! the Track Edit panel only shows which file is open, so what is held here is
//! one open forest per loaded node.
//!
//! The viewer never builds one behind the person's back. Opening is free --
//! a `.kdf` is opened without decoding a tree or a descriptor block
//! (`specs/core/features/lazy-kdforest-query.md`) -- so the default path is
//! opened on sight when the file is there; building one reads every `.sift` of
//! the capture and is a background task the person asks for.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use sfmtool_core::features::kdforest::{
    KdForestParams, KdForestU8, KdfSiftSources, KdfWorkspaceContents, KdfWorkspaceMetadata,
    KdfWriteOptions, LazyKdForestOptions, LazyKdForestU8,
};
use sfmtool_core::progress::Progress;
use sfmtool_core::{progress_note, SfmrReconstruction};

use crate::action_log::{Actor, Kind};
use crate::background::{Finished, Job, Operation};
use crate::scene::ReconId;
use crate::state::AppState;

#[cfg(test)]
pub(crate) mod tests;

/// What a workspace's descriptor index is called.
pub(crate) const INDEX_FILE_NAME: &str = "index.kdf";

/// One node's open descriptor index.
#[derive(Clone)]
pub(crate) struct DescriptorIndex {
    /// The `.kdf` it was opened from, as the panel and the wire name it.
    pub(crate) path: PathBuf,
    /// The forest itself.
    ///
    /// Behind an [`Arc`] because a search runs on a **worker**: the forest owns
    /// a query thread pool and a bounded decoded-block cache, so a task takes a
    /// clone of the handle rather than opening the file a second time and
    /// paying for a second cache.
    pub(crate) forest: Arc<LazyKdForestU8>,
}

impl DescriptorIndex {
    /// How many descriptors the corpus holds.
    pub(crate) fn feature_count(&self) -> usize {
        self.forest.len()
    }
}

/// Where a node's `.sift` files live, which is where its index lives too.
///
/// Features are stored beside the images they came from, under the workspace's
/// feature prefix directory, so a capture whose images sit in one directory has
/// one such directory and the index goes in it -- the answer
/// `SfmrReconstruction::sift_path_for_image` already resolves for image 0. A
/// capture spread over several image directories has several, and the first
/// image's is the one taken: an index is over the whole capture whichever of
/// them holds it, and a rule that picked differently per call would leave a
/// second session looking in a different place.
///
/// **The separators are this platform's, all of them.** The feature directory
/// is stored in the `.sfmr` with `/` between its parts, whatever wrote it, so a
/// path joined from it on Windows comes out spelled
/// `…\images\features/sift-sfmtool-…\index.kdf` -- which opens, and reads as
/// two conventions arguing in a panel, a reply and a log row. Rebuilding the
/// path from its components is what settles it: `Path::components` splits on
/// both separators and `PathBuf` re-joins with the platform's own.
pub(crate) fn default_index_path(recon: &SfmrReconstruction) -> Option<PathBuf> {
    if recon.image_table.images.is_empty() {
        return None;
    }
    let sift = recon.sift_path_for_image(0);
    Some(normalized(&sift.parent()?.join(INDEX_FILE_NAME)))
}

/// `path` spelled with this platform's separator throughout.
fn normalized(path: &Path) -> PathBuf {
    path.components().collect()
}

/// `path` as the index of `recon` may be written to, or why it may not.
///
/// A caller naming the file is worth keeping -- a second index over the same
/// capture, under a name of its own, is a reasonable thing for an agent to ask
/// for -- and what it is not worth is writing anywhere on the disk. So the path
/// is resolved against the node's workspace directory when it is relative,
/// lexically normalised (`.` and `..` folded, no filesystem walked, since the
/// file is not there yet), and refused when the answer leaves that tree.
///
/// Lexical rather than canonical on purpose: `canonicalize` needs the path to
/// exist, and this one is about to be created. A symlink out of the workspace
/// therefore passes, which is a thing the person who made the symlink asked
/// for. A reconstruction that names no workspace directory has no tree to be
/// outside of, and every path is allowed.
fn within_workspace(recon: &SfmrReconstruction, path: &Path) -> Result<PathBuf, String> {
    let workspace = lexically_normalized(&recon.workspace_dir);
    let joined = if path.is_absolute() {
        path.to_path_buf()
    } else {
        recon.workspace_dir.join(path)
    };
    let resolved = lexically_normalized(&joined);
    if workspace.as_os_str().is_empty() || resolved.starts_with(&workspace) {
        return Ok(resolved);
    }
    Err(format!(
        "{} is outside the workspace at {}, and an index is written beside the \
         features it indexes.",
        resolved.display(),
        workspace.display()
    ))
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
    /// The descriptor index open beside `id`, or `None` when none is.
    pub(crate) fn descriptor_index(&self, id: ReconId) -> Option<&DescriptorIndex> {
        self.descriptor_indexes.get(&id)?.as_ref()
    }

    /// Where `id`'s index goes when nobody names a path.
    pub(crate) fn default_descriptor_index_path(&self, id: ReconId) -> Option<PathBuf> {
        default_index_path(self.node(id)?.recon())
    }

    /// Open the default index of `id` if the file is there and none is open.
    ///
    /// What the Track Edit panel and the first item put on a node's bench call:
    /// a `.kdf` opens without decoding a tree or a descriptor block, so looking
    /// costs a stat and a header read, and a session that finds the file the
    /// last one built is a session that can search. It **builds nothing** and
    /// says nothing when there is no file: an index that is not there is the
    /// normal state of a workspace, not a refusal.
    pub(crate) fn open_default_descriptor_index(&mut self, id: ReconId) {
        if self.descriptor_indexes.contains_key(&id) {
            return;
        }
        let path = self
            .default_descriptor_index_path(id)
            .filter(|p| p.is_file());
        let Some(path) = path else {
            // Remembered as a miss, so the panel's every frame is not a stat of
            // a file the workspace does not have.
            self.descriptor_indexes.insert(id, None);
            return;
        };
        // **The row this writes is the viewer's.** Nobody asked for it: the look
        // happens because something was put on the bench, in the middle of the
        // step that put it there. Attributed to whoever was acting, it would be
        // the last row that step wrote, and a wire reply that reports the
        // step's own sentence would report this one instead.
        let standing = self.action_log.actor();
        self.action_log.set_actor(Actor::Viewer);
        let opened = self.open_descriptor_index(id, Some(path));
        if let Err(why) = opened {
            // Said out loud: a file sitting where the index goes and not being
            // usable is worth a row, and this is the one chance to write it --
            // the miss is remembered, so nothing looks again.
            self.descriptor_indexes.insert(id, None);
            self.action_log.fail(Kind::Bench, why);
        }
        self.action_log.set_actor(standing);
    }

    /// Open `path`, or the default when none is named, as `id`'s descriptor
    /// index.
    ///
    /// **The corpus has to be this node's images, in this node's order.** A
    /// match names a corpus image index and the candidate it becomes names a
    /// node image index; nothing in the search can tell the two apart, so the
    /// check is here, once, where a file is adopted. An index whose image table
    /// says something else is refused naming the first image it disagrees on,
    /// rather than seeding observations into the wrong photographs.
    pub(crate) fn open_descriptor_index(
        &mut self,
        id: ReconId,
        path: Option<PathBuf>,
    ) -> Result<PathBuf, String> {
        let node = self
            .node(id)
            .ok_or_else(|| "That reconstruction is no longer loaded.".to_string())?;
        let path = match path {
            Some(path) => path,
            None => default_index_path(node.recon()).ok_or_else(|| {
                "This reconstruction has no images, so it has no feature directory to \
                 look in."
                    .to_string()
            })?,
        };
        let forest = LazyKdForestU8::open(&path, LazyKdForestOptions::default())
            .map_err(|e| format!("Cannot open {}: {e}", path.display()))?;
        check_images_match(&forest, node.recon(), &path)?;
        let label = node.label.clone();
        let features = forest.len();
        self.descriptor_indexes.insert(
            id,
            Some(DescriptorIndex {
                path: path.clone(),
                forest: Arc::new(forest),
            }),
        );
        self.action_log.record(
            Kind::Bench,
            format!(
                "Opened the descriptor index of {label}: {} with {features} descriptors",
                path.display()
            ),
        );
        Ok(path)
    }

    /// Why `id` cannot have an index built for it, or `None` when it can.
    ///
    /// Asked by the step itself, so the button and the task cannot disagree
    /// about when there is something to index. The panel asks
    /// [`Self::descriptor_sources_refusal`] instead and puts the busy question
    /// in front of it itself, because this half stats a file per image of the
    /// capture and a panel that asked it per frame would do that per frame.
    pub(crate) fn build_descriptor_index_refusal(&self, id: ReconId) -> Option<String> {
        self.busy_refusal(id)
            .or_else(|| self.descriptor_sources_refusal(id))
    }

    /// Why there is nothing to index beside `id`, or `None` when there is.
    ///
    /// **Stats a `.sift` path per image**, so a caller that asks repeatedly
    /// caches the answer: what it is a function of is which files exist beside
    /// the workspace, and nothing in the viewer changes that.
    pub(crate) fn descriptor_sources_refusal(&self, id: ReconId) -> Option<String> {
        let node = self.node(id)?;
        readable_sift_files(node.recon()).is_empty().then(|| {
            format!(
                "No .sift file of {} could be found, so there are no descriptors to index.",
                node.label
            )
        })
    }

    /// Build a descriptor index over every `.sift` file of `id`, write it, and
    /// open it, on a worker thread.
    ///
    /// The corpus carries **one image-table row per image of the node**, in the
    /// node's own order, including the images that have no `.sift` file: an
    /// image with no features contributes no descriptor and still takes its
    /// row, which is what keeps a corpus image index and a node image index the
    /// same number.
    ///
    /// `progress` names three phases: `read descriptors`, `build forest` and
    /// `write index`.
    pub(crate) fn start_build_descriptor_index(
        &mut self,
        id: ReconId,
        path: Option<PathBuf>,
    ) -> Result<(), String> {
        let outcome = self.begin_build_descriptor_index(id, path);
        if let Err(message) = &outcome {
            self.action_log.fail(Kind::Bench, message.clone());
        }
        outcome
    }

    fn begin_build_descriptor_index(
        &mut self,
        id: ReconId,
        path: Option<PathBuf>,
    ) -> Result<(), String> {
        if let Some(why) = self.build_descriptor_index_refusal(id) {
            return Err(why);
        }
        let node = self
            .node(id)
            .ok_or_else(|| "That reconstruction is no longer loaded.".to_string())?;
        let recon = node.recon();
        let path = match path {
            Some(path) => within_workspace(recon, &path)?,
            None => default_index_path(recon).ok_or_else(|| {
                "This reconstruction has no images, so it has no feature directory to \
                 write into."
                    .to_string()
            })?,
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
        let job: Job = Box::new(move |progress| build(plan, progress));
        self.start_background_task(Operation::BUILD_DESCRIPTOR_INDEX, id, job)
    }

    /// Install an index a background build produced.
    pub(crate) fn install_descriptor_index(
        &mut self,
        id: ReconId,
        path: PathBuf,
        forest: Arc<LazyKdForestU8>,
    ) {
        self.descriptor_indexes
            .insert(id, Some(DescriptorIndex { path, forest }));
    }

    /// Forget the index of a node that has left the scene.
    pub(crate) fn forget_descriptor_index(&mut self, id: ReconId) {
        self.descriptor_indexes.remove(&id);
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

    let mut descriptors: Vec<u8> = Vec::new();
    let mut origins = Vec::new();
    let mut geometry = Vec::new();
    let mut dimension = 0usize;
    {
        let mut phase = progress.phase("read descriptors");
        for (index, (image, sift_path)) in sources.iter().enumerate() {
            phase.set_fraction(index as f32 / sources.len() as f32);
            let data = match sfmtool_sift_format::read_sift_partial(sift_path, usize::MAX) {
                Ok(data) => data,
                Err(e) => {
                    return Finished::Failed(format!("Cannot read {}: {e}", sift_path.display()))
                }
            };
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
        let _phase = progress.phase("build forest");
        KdForestU8::build(&descriptors, count, dimension, KdForestParams::default())
    };

    let images = image_names.len();
    let sources = KdfSiftSources {
        workspace,
        image_names,
        // The digests are what a verification would check a `.sift` file
        // against, and the viewer holds neither: the index is opened beside the
        // very reconstruction it was built from, and the image-table check at
        // open time is what says the two belong together.
        feature_tool_hashes: vec![[0u8; 16]; images],
        sift_content_hashes: vec![[0u8; 16]; images],
        origins,
        geometry,
    };
    {
        let _phase = progress.phase("write index");
        if let Some(parent) = path.parent() {
            if let Err(e) = std::fs::create_dir_all(parent) {
                return Finished::Failed(format!("Cannot create {}: {e}", parent.display()));
            }
        }
        // A rebuild replaces what is there: the writer refuses an existing file,
        // and the person asked for this index rather than for a second one.
        let _ = std::fs::remove_file(&path);
        if let Err(e) = forest.write_kdf(&path, Some(&sources), &KdfWriteOptions::default()) {
            return Finished::Failed(format!("Cannot write {}: {e}", path.display()));
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
    Finished::DescriptorIndex {
        text: format!(
            "Built the descriptor index {}: {count} descriptors of {} images",
            path.display(),
            images
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

/// Refuse an index whose corpus is not this reconstruction's images.
fn check_images_match(
    forest: &LazyKdForestU8,
    recon: &SfmrReconstruction,
    path: &Path,
) -> Result<(), String> {
    let table = forest
        .image_table()
        .map_err(|e| format!("Cannot read the image table of {}: {e}", path.display()))?
        .ok_or_else(|| {
            format!(
                "{} carries no image table, so nothing says which photograph a hit \
                 came from.",
                path.display()
            )
        })?;
    let images = &recon.image_table.images;
    if table.names.len() != images.len() {
        return Err(format!(
            "{} indexes {} images and this reconstruction has {}.",
            path.display(),
            table.names.len(),
            images.len()
        ));
    }
    for (index, (indexed, image)) in table.names.iter().zip(images.iter()).enumerate() {
        if indexed != &image.name {
            return Err(format!(
                "{} has {indexed:?} as image {index} and this reconstruction has {:?}; \
                 an index has to be over this reconstruction's own images, in its own order.",
                path.display(),
                image.name
            ));
        }
    }
    Ok(())
}
