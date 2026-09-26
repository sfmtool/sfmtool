// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The index files of a reconstruction: the two files beside its `.sfmr` that
//! a bench search reads, and the one background operation that builds both.
//!
//! See `specs/gui/index-files.md`. The two are the node's SIFT index,
//! `<stem>-sift-index.kdf` ([`crate::sift_index`]), and its cluster-patches
//! file, `<stem>-cluster-patches.matches` ([`crate::cluster_patches`]). The
//! second is made from the first, so they are built by one operation, *Build
//! Index Files*, and each keeps its own none / current / stale state beside
//! the node. This module holds what belongs to the pair: the state word both
//! use, the refusals of the build, the build's plan and job, and the look that
//! opens both on sight.

use std::path::PathBuf;
use std::sync::Arc;

use sfmtool_core::features::kdforest::LazyKdForestU8;
use sfmtool_core::progress::Progress;

use crate::background::{Finished, IndexFilesEnd, Job, Operation};
use crate::cluster_patches::{self, ClusterPlan};
use crate::scene::{ReconId, SceneNode};
use crate::sift_index::{self, BuildPlan};
use crate::state::AppState;

#[cfg(test)]
pub(crate) mod tests;

/// Which of the three states one of a node's index files is in.
///
/// `none` is a node with no file at that path and none opened by hand;
/// `current` is a file that answers for the node as it stands; anything else
/// is `stale`. What "answers for the node" means is each file's own
/// ([`crate::sift_index`], [`crate::cluster_patches`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum IndexFileState {
    None,
    Current,
    Stale,
}

impl IndexFileState {
    /// The word the rows and the wire all use.
    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Current => "current",
            Self::Stale => "stale",
        }
    }
}

/// Why a stage of the build handed back nothing, which is the two ways it can
/// end without its file.
pub(crate) enum Stopped {
    Cancelled,
    Failed(String),
}

/// What a node with no path on disk is told when it is asked for its search
/// files.
pub(crate) fn unsaved_refusal(node: &SceneNode) -> String {
    format!(
        "Save {} first: the index files are written beside the .sfmr file.",
        node.label
    )
}

/// What the entry that builds a node's index files is called when the node
/// has neither file, in every menu that offers it and the tests that aim at
/// them.
pub(crate) const BUILD_INDEX_FILES: &str = "Build Index Files";

/// What the same entry is called when the node has either file.
pub(crate) const REBUILD_INDEX_FILES: &str = "Rebuild Index Files";

/// How the build's time divides between the index and the cluster patches,
/// when it builds both.
///
/// On the 17-image seoul_bull capture, 37K descriptors, the index is 0.09 s of
/// a 1.1 s build and the clustering and the refinement are the rest, so the
/// index takes an eighth. An exact binary fraction, so the two shares sum to
/// one in `f32` and a finished bar reaches its end.
const INDEX_SHARE: f32 = 0.125;

/// The share of the build that is the cluster patches.
const CLUSTER_SHARE: f32 = 0.875;

impl AppState {
    /// Look for `id`'s index files if nothing has yet, and re-derive their
    /// states when the node's image table, or the index the cluster patches
    /// are judged against, has moved.
    ///
    /// What Track View, the Scene tree and the first item put on a node's bench
    /// call. It builds nothing and says nothing when there is no file.
    pub(crate) fn refresh_index_files(&mut self, id: ReconId) {
        self.refresh_sift_index(id);
        self.refresh_cluster_patches(id);
    }

    /// Whether the operation running on `id` is the index-files build.
    pub(crate) fn building_index_files(&self, id: ReconId) -> bool {
        self.background_task().is_some_and(|task| {
            task.node == Some(id) && task.operation.name == Operation::BUILD_INDEX_FILES.name
        })
    }

    /// Why there is nowhere to put `id`'s index files, or `None` when there
    /// is.
    pub(crate) fn index_files_home_refusal(&self, id: ReconId) -> Option<String> {
        let node = self.node(id)?;
        node.path.is_none().then(|| unsaved_refusal(node))
    }

    /// Why `id` cannot have its index files built, or `None` when it can.
    ///
    /// Asked by the step itself, so the menu entry and the task cannot disagree
    /// about when there is something to build from.
    pub(crate) fn build_index_files_refusal(&self, id: ReconId) -> Option<String> {
        self.busy_refusal(id)
            .or_else(|| self.index_files_home_refusal(id))
            .or_else(|| self.sift_sources_refusal(id))
    }

    /// What the build entry is called for `id`: a first build when the node
    /// has neither file, a rebuild when it has either.
    pub(crate) fn index_files_build_label(&self, id: ReconId) -> &'static str {
        let none = self.sift_index_state(id) == IndexFileState::None
            && self.cluster_patches_state(id) == IndexFileState::None;
        match none {
            true => BUILD_INDEX_FILES,
            false => REBUILD_INDEX_FILES,
        }
    }

    /// Build `id`'s index files on a worker thread: its SIFT index, then its
    /// cluster-patches file from that index, each written at the node's own
    /// path beside the `.sfmr` and opened.
    ///
    /// **A current index is kept when only the cluster patches are not
    /// current.** When the index open beside the node is current, is at the
    /// node's own index path, and the cluster-patches file is not current, the
    /// build skips the index and makes the cluster patches from the index that
    /// is open. Every other build writes both, so a node whose two files are
    /// current rebuilds both.
    pub(crate) fn start_build_index_files(&mut self, id: ReconId) -> Result<(), String> {
        let outcome = self
            .build_index_files_job(id)
            .and_then(|job| self.start_background_task(Operation::BUILD_INDEX_FILES, id, job));
        if let Err(message) = &outcome {
            self.action_log
                .fail(crate::action_log::Kind::Bench, message.clone());
        }
        outcome
    }

    /// The work a build does, as a closure that owns everything it reads.
    ///
    /// Separated from starting it for the reason every other operation's job is:
    /// the refusals and the plan are the state's to work out, and what crosses
    /// to the worker is one closure holding no reference into the scene.
    pub(crate) fn build_index_files_job(&self, id: ReconId) -> Result<Job, String> {
        if let Some(why) = self.build_index_files_refusal(id) {
            return Err(why);
        }
        let node = self
            .node(id)
            .ok_or_else(|| crate::state::NOT_LOADED.to_string())?;
        let own = sift_index::index_path(node);
        let keep = self.sift_index_state(id) == IndexFileState::Current
            && self.cluster_patches_state(id) != IndexFileState::Current;
        let index = match self.sift_index(id) {
            Some(open) if keep && Some(&open.path) == own.as_ref() => IndexSource::Keep {
                path: open.path.clone(),
                forest: Arc::clone(&open.forest),
            },
            _ => IndexSource::Build(BuildPlan::of(node)?),
        };
        let plan = IndexFilesPlan {
            label: node.label.clone(),
            index,
            cluster: ClusterPlan::of(node)?,
        };
        Ok(Box::new(move |progress| build(plan, progress)))
    }

    /// Install what a index-files build produced: the index it wrote, if it
    /// wrote one, and the cluster-patches file, if it wrote that.
    ///
    /// Each is classified by the same function an opened file goes through,
    /// rather than taken as current because the build just made it.
    pub(crate) fn install_index_files(
        &mut self,
        id: ReconId,
        index: Option<(PathBuf, Arc<LazyKdForestU8>)>,
        cluster_patches: Option<PathBuf>,
    ) {
        if let Some((path, forest)) = index {
            self.install_sift_index(id, path, forest);
        }
        if let Some(path) = cluster_patches {
            self.install_cluster_patches(id, path);
        }
    }

    /// Open `id`'s index files: the named files, or the node's own where a
    /// path is not named.
    ///
    /// A named file has to open, and the step is refused naming it when it
    /// does not. A file that is not named is opened when it is at the node's
    /// own path, and reads `none` when it is not there. Opening again is how a
    /// person asks about files that changed on disk under a running viewer,
    /// since the viewer does not watch them. Refused when neither file opened.
    pub(crate) fn open_index_files(
        &mut self,
        id: ReconId,
        sift_index: Option<PathBuf>,
        cluster_patches: Option<PathBuf>,
    ) -> Result<(), String> {
        let node = self
            .node(id)
            .ok_or_else(|| crate::state::NOT_LOADED.to_string())?;
        let label = node.label.clone();
        let named_index = sift_index.is_some();
        let named_patches = cluster_patches.is_some();
        let index = sift_index.or_else(|| sift_index::index_path(node).filter(|p| p.is_file()));
        let patches = cluster_patches
            .or_else(|| cluster_patches::cluster_patches_path(node).filter(|p| p.is_file()));
        if index.is_none() && patches.is_none() {
            return Err(match self.index_files_home_refusal(id) {
                Some(why) => why,
                None => format!("No index file of {label} is there to open."),
            });
        }
        // The index first: the cluster patches are judged against it.
        match index {
            Some(path) => {
                if let Err(why) = self.open_sift_index(id, path) {
                    if named_index {
                        return Err(why);
                    }
                    self.sift_indexes.insert(id, None);
                    self.action_log.fail(crate::action_log::Kind::Bench, why);
                }
            }
            None => {
                self.sift_indexes.insert(id, None);
            }
        }
        match patches {
            Some(path) => {
                if let Err(why) = self.open_cluster_patches(id, path) {
                    if named_patches {
                        return Err(why);
                    }
                    self.cluster_patches.insert(id, None);
                    self.action_log.fail(crate::action_log::Kind::Bench, why);
                }
            }
            None => {
                self.cluster_patches.insert(id, None);
            }
        }
        Ok(())
    }

    /// Let go of both of `id`'s index files, leaving the files where they
    /// are.
    ///
    /// Remembered as misses, so the next frame that draws the rows does not
    /// open them again. Refused when neither is open.
    pub(crate) fn close_index_files(&mut self, id: ReconId) -> Result<(), String> {
        let node = self
            .node(id)
            .ok_or_else(|| crate::state::NOT_LOADED.to_string())?;
        let label = node.label.clone();
        let paths: Vec<String> = [
            self.sift_index(id)
                .map(|index| index.path.display().to_string()),
            self.cluster_patches(id)
                .map(|file| file.path.display().to_string()),
        ]
        .into_iter()
        .flatten()
        .collect();
        if paths.is_empty() {
            return Err(format!("No index file is open beside {label}."));
        }
        self.sift_indexes.insert(id, None);
        self.cluster_patches.insert(id, None);
        self.action_log.record(
            crate::action_log::Kind::Bench,
            format!("Closed the index files of {label}: {}", paths.join(", ")),
        );
        Ok(())
    }

    /// Forget both files of a node that has left the scene.
    pub(crate) fn forget_index_files(&mut self, id: ReconId) {
        self.forget_sift_index(id);
        self.forget_cluster_patches(id);
    }
}

/// Where the index the cluster patches are made from comes from.
enum IndexSource {
    /// Built by this job, and then opened.
    Build(BuildPlan),
    /// The current index already open beside the node.
    Keep {
        path: PathBuf,
        forest: Arc<LazyKdForestU8>,
    },
}

/// Everything the build needs, owned.
struct IndexFilesPlan {
    /// The node's label, for the sentences.
    label: String,
    index: IndexSource,
    cluster: ClusterPlan,
}

/// The build: the index when it is not kept, then the cluster patches from it.
///
/// A build that wrote its index and then stopped or failed in the cluster
/// patches still hands the index back, so the node opens the file that is now
/// on disk rather than going on showing the one it replaced.
fn build(plan: IndexFilesPlan, progress: &Progress<'_>) -> Finished {
    let IndexFilesPlan {
        label,
        index,
        cluster,
    } = plan;
    let (built, index_path, forest, cluster_share) = match index {
        IndexSource::Keep { path, forest } => (None, path, forest, *progress),
        IndexSource::Build(index_plan) => {
            let [index_share, cluster_share] = progress.split([INDEX_SHARE, CLUSTER_SHARE]);
            let built = match sift_index::build_index(index_plan, &index_share) {
                Ok(built) => built,
                Err(Stopped::Cancelled) => return Finished::Cancelled,
                Err(Stopped::Failed(why)) => return Finished::Failed(why),
            };
            let text = format!(
                "the SIFT index {} with {} descriptors of {} images",
                built.path.display(),
                built.descriptors,
                built.images
            );
            let forest = Arc::new(built.forest);
            (
                Some((built.path.clone(), Arc::clone(&forest), text)),
                built.path,
                forest,
                cluster_share,
            )
        }
    };
    let made = cluster_patches::build(cluster, &forest, &index_path, &cluster_share);
    let index_text = built.as_ref().map(|(_, _, text)| text.clone());
    let index = built.map(|(path, forest, _)| (path, forest));
    match made {
        Ok(summary) => {
            let patches = format!(
                "the cluster patches {} with {} clusters of {} members",
                summary.path.display(),
                summary.clusters,
                summary.members
            );
            let text = match &index_text {
                Some(index_text) => {
                    format!("Built the index files of {label}: {index_text}, and {patches}")
                }
                None => format!(
                    "Built the index files of {label}: {patches}, from the SIFT index {} \
                     that was already current",
                    index_path.display()
                ),
            };
            Finished::IndexFiles {
                index,
                cluster_patches: Some(summary.path),
                end: IndexFilesEnd::Built(text),
            }
        }
        Err(Stopped::Cancelled) => Finished::IndexFiles {
            index,
            cluster_patches: None,
            end: IndexFilesEnd::Cancelled,
        },
        Err(Stopped::Failed(why)) => Finished::IndexFiles {
            end: IndexFilesEnd::Failed(match &index_text {
                Some(index_text) => {
                    format!("Built {index_text}, and could not build the cluster patches: {why}")
                }
                None => format!("Could not build the cluster patches: {why}"),
            }),
            index,
            cluster_patches: None,
        },
    }
}
