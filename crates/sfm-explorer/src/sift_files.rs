// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The `.sift` files of a reconstruction's images, and the one background
//! operation that builds them, *Build SIFT Files*.
//!
//! See `specs/gui/index-files.md` § "Building the .sift files". The files are
//! the workspace's, at `<workspace>/<image dir>/<feature_prefix_dir>/<image
//! name>.sift`, where `SfmrReconstruction::sift_path_for_image` and so *Build
//! Index Files* look for them. The build is
//! `sfmtool_core::features::sift::build_sift_files`, run with the feature
//! options the node records in its `.sfmr` workspace definition. It writes
//! `.sift` version 2 files, which never come from version 1 settings or go
//! beside version 1 files, and it can only extract with the `sfmtool` feature
//! tool. So the entry first changes the definition, in the viewer's memory
//! only, in two cases: a definition naming another tool is switched to the
//! `sfmtool` tool with its default options, and a version 1 `sfmtool`
//! definition (one with no `image_to_gray`) is converted to version 2 as
//! `sfm ws upgrade` converts a workspace
//! (`sfmtool_core::workspace::upgrade_feature_options`), with a new folder
//! named from the converted options. Either change is pushed as a version
//! when the build succeeds.

use std::path::PathBuf;
use std::sync::Arc;

use sfmtool_core::features::sift::{
    build_sift_files, BuildSiftFilesError, BuildSiftFilesOptions, SFMTOOL_FEATURE_TOOL,
    SFMTOOL_FEATURE_TYPE,
};
use sfmtool_core::progress::Progress;
use sfmtool_core::workspace::{feature_prefix_dir, has_image_to_gray, upgrade_feature_options};
use sfmtool_core::{EditedReconstruction, ObservationSource, PointMap};

use crate::background::{Finished, Job, Operation};
use crate::scene::{ReconId, SceneNode};
use crate::state::AppState;

#[cfg(test)]
pub(crate) mod tests;

/// What the entry that builds a node's `.sift` files is called when the node's
/// workspace definition already extracts with the `sfmtool` tool, in every
/// menu that offers it and the tests that aim at them.
pub(crate) const BUILD_SIFT_FILES: &str = "Build SIFT Files";

/// What the same entry is called when the definition names another tool, which
/// the entry switches first.
pub(crate) const SWITCH_AND_BUILD_SIFT_FILES: &str =
    "Switch SIFT Extraction to sfmtool and Build SIFT Files";

/// How a build changes the node's workspace definition before it extracts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum RecordChange {
    /// None: the definition is a version 2 `sfmtool` one, and the files go in
    /// its folder.
    Keep,
    /// The definition names this other tool, and is switched to the
    /// `sfmtool` defaults.
    Switch(String),
    /// The definition is a version 1 `sfmtool` one, and is converted to
    /// version 2, with a new folder.
    Upgrade,
}

/// What a build of `node` does to its definition first.
fn record_change(node: &SceneNode) -> RecordChange {
    let contents = &node.recon().metadata.workspace.contents;
    if !extracts_with_sfmtool(node) {
        RecordChange::Switch(contents.feature_tool.clone())
    } else if !has_image_to_gray(&contents.feature_options) {
        RecordChange::Upgrade
    } else {
        RecordChange::Keep
    }
}

/// What the entry's hover says, in every menu that offers it: what the build
/// does, and first, when the build changes the node's definition, what the
/// change does.
pub(crate) fn build_sift_files_hover(change: &RecordChange) -> String {
    let build = "Extract SIFT features from every photograph of this reconstruction with the \
                 sfmtool feature tool, and write each image's .sift file in the workspace, \
                 where Build Index Files reads them. An image whose .sift file is up to date is \
                 skipped. Runs on a worker thread and can be cancelled.";
    match change {
        RecordChange::Keep => build.to_string(),
        RecordChange::Upgrade => format!(
            "This reconstruction records version 1 sfmtool settings, and no .sift file is \
             written from version 1 settings. First convert the record to version 2 as \
             'sfm ws upgrade' converts a workspace, in this reconstruction only: every option \
             is kept, its gray_formula becomes image_to_gray, and the files go in a new \
             folder named from the converted options. The workspace's .sfm-workspace.json is \
             not changed. Then: {build} The conversion is a version once every file is \
             built; Undo (Ctrl+Z) puts the old record back, and a save writes the new one \
             into the .sfmr."
        ),
        RecordChange::Switch(from) => format!(
            "This reconstruction records '{from}' as its SIFT extraction, and only the sfmtool \
             tool can build .sift files here. First switch the record to sfmtool with its \
             default options, in this reconstruction only: the workspace's \
             .sfm-workspace.json is not changed. Then: {build} The switch is a version once \
             every file is built; Undo (Ctrl+Z) puts the old record back, and a save writes \
             the new one into the .sfmr."
        ),
    }
}

/// Whether `node`'s workspace definition extracts with the `sfmtool` tool, the
/// one tool the viewer can build `.sift` files with.
fn extracts_with_sfmtool(node: &SceneNode) -> bool {
    node.recon()
        .metadata
        .workspace
        .contents
        .feature_tool
        .eq_ignore_ascii_case(SFMTOOL_FEATURE_TOOL)
}

impl AppState {
    /// What the build entry is called for `id`: a plain build when its
    /// workspace definition extracts with `sfmtool`, and a switch and a build
    /// when it names another tool.
    pub(crate) fn sift_files_build_label(&self, id: ReconId) -> &'static str {
        match self.sift_extraction_switch_from(id) {
            None => BUILD_SIFT_FILES,
            Some(_) => SWITCH_AND_BUILD_SIFT_FILES,
        }
    }

    /// The feature tool `id`'s workspace definition names when a build would
    /// switch it to `sfmtool`, or `None` when it already extracts with
    /// `sfmtool` or is not loaded.
    pub(crate) fn sift_extraction_switch_from(&self, id: ReconId) -> Option<String> {
        match record_change(self.node(id)?) {
            RecordChange::Switch(from) => Some(from),
            _ => None,
        }
    }

    /// What a build of `id` does to its definition first, `Keep` when it is
    /// not loaded.
    pub(crate) fn sift_record_change(&self, id: ReconId) -> RecordChange {
        self.node(id).map_or(RecordChange::Keep, record_change)
    }

    /// Why `id` cannot have its `.sift` files built, or `None` when it can.
    ///
    /// Cheap enough to ask from a menu every frame: one stat of the workspace
    /// directory. Whether the files are already there is not asked, because a
    /// build skips every image whose `.sift` file is up to date.
    pub(crate) fn build_sift_files_refusal(&self, id: ReconId) -> Option<String> {
        if let Some(why) = self.busy_refusal(id) {
            return Some(why);
        }
        let node = self.node(id)?;
        let recon = node.recon();
        let label = &node.label;
        if recon.image_set.images.is_empty() {
            return Some(format!(
                "{label} has no images, so there are no .sift files to build."
            ));
        }
        if !recon.workspace_dir.is_dir() {
            return Some(format!(
                "The workspace directory '{}' of {label} is not there, so its photographs \
                 cannot be read.",
                recon.workspace_dir.display()
            ));
        }
        // A switch moves where the `.sift` files are looked for. Feature indexes
        // name features of the files the reconstruction was made from, and the
        // files at the new place hold other features.
        let sift_files = matches!(
            recon.point_set.observations,
            ObservationSource::SiftFiles { .. }
        );
        let contents = &recon.metadata.workspace.contents;
        match record_change(node) {
            RecordChange::Switch(_) if sift_files => {
                return Some(format!(
                    "The observations of {label} are feature indexes into its '{}' .sift \
                     files, so its SIFT extraction cannot switch to sfmtool: the new files \
                     would hold other features. Convert it to embedded patches first.",
                    contents.feature_tool
                ));
            }
            RecordChange::Upgrade if sift_files => {
                return Some(format!(
                    "The observations of {label} are feature indexes into its version 1 .sift \
                     files, so its record cannot move to a version 2 folder: the feature \
                     indexes name features of the old files, which a new folder would not \
                     hold. Convert it to embedded patches first."
                ));
            }
            RecordChange::Upgrade => {
                if let Err(e) =
                    upgrade_feature_options(&contents.feature_tool, &contents.feature_options)
                {
                    return Some(format!(
                        "The version 1 SIFT extraction record of {label} cannot be converted to \
                         version 2: {e}."
                    ));
                }
            }
            _ => {}
        }
        None
    }

    /// Build `id`'s `.sift` files on a worker thread, switching its workspace
    /// definition to the `sfmtool` tool first when it names another.
    pub(crate) fn start_build_sift_files(&mut self, id: ReconId) -> Result<Option<u64>, String> {
        let outcome = self.build_sift_files_job(id).and_then(|job| {
            self.start_background_task(Operation::BUILD_SIFT_FILES, id, job)
                .map(Some)
        });
        if let Err(message) = &outcome {
            self.action_log
                .fail(crate::action_log::Kind::Bench, message.clone());
        }
        outcome
    }

    /// The work a build does, as a closure that owns everything it reads.
    ///
    /// The images are `<workspace>/<image name>`, and the options are the
    /// node's own workspace definition when it is a version 2 `sfmtool` one,
    /// that definition converted to version 2 with a new folder when it is a
    /// version 1 `sfmtool` one, or the `sfmtool` defaults when the definition
    /// names another tool. None reads the workspace's `.sfm-workspace.json`:
    /// the files go where the node looks for them, after the change.
    pub(crate) fn build_sift_files_job(&self, id: ReconId) -> Result<Job, String> {
        if let Some(why) = self.build_sift_files_refusal(id) {
            return Err(why);
        }
        let node = self
            .node(id)
            .ok_or_else(|| crate::state::NOT_LOADED.to_string())?;
        let recon = node.recon();
        let own = &recon.metadata.workspace.contents;
        let change = record_change(node);
        let (options, switch) = match change {
            RecordChange::Keep => (
                BuildSiftFilesOptions {
                    feature_options: own.feature_options.clone(),
                    feature_prefix_dir: PathBuf::from(&own.feature_prefix_dir),
                    concurrent_images: None,
                },
                None,
            ),
            // The conversion `sfm ws upgrade` makes, with a folder named from
            // the converted options, as every folder made under version 2 is.
            RecordChange::Upgrade => {
                let feature_options =
                    upgrade_feature_options(&own.feature_tool, &own.feature_options)
                        .map_err(|e| e.to_string())?;
                let folder = feature_prefix_dir(
                    SFMTOOL_FEATURE_TOOL,
                    SFMTOOL_FEATURE_TYPE,
                    &feature_options,
                );
                (
                    BuildSiftFilesOptions {
                        feature_options,
                        feature_prefix_dir: PathBuf::from(folder),
                        concurrent_images: None,
                    },
                    Some(change),
                )
            }
            // After a switch, the definition `sfm ws init` writes for the tool:
            // its default options and the prefix directory named from them.
            RecordChange::Switch(_) => (BuildSiftFilesOptions::sfmtool_defaults(), Some(change)),
        };
        let switch = switch.map(|change| Switch {
            change,
            old_folder: own.feature_prefix_dir.clone(),
            edited: node.history.current().clone(),
        });
        let plan = SiftFilesPlan {
            label: node.label.clone(),
            images: recon
                .image_set
                .images
                .iter()
                .map(|image| recon.workspace_dir.join(&image.name))
                .collect(),
            options,
            switch,
        };
        Ok(Box::new(move |progress| build(plan, progress)))
    }

    /// What a node's caches say about its `.sift` files, asked again after a
    /// build wrote some, however it ended.
    ///
    /// An open SIFT index is classified again, since whether it is current
    /// depends on the `.sift` files' content hashes, and that asks the cluster
    /// patches again too. The panels' caches are dropped by the frame, which
    /// reads the node off `Polled::installed`.
    pub(crate) fn sift_files_built(&mut self, id: ReconId) {
        if let Some(index) = self.sift_index(id) {
            let (path, forest) = (index.path.clone(), Arc::clone(&index.forest));
            self.install_sift_index(id, path, forest);
        }
    }
}

/// A change of the node's workspace definition, a switch to the `sfmtool` tool
/// or a conversion to version 2, pushed as a version when the build succeeds.
struct Switch {
    /// Which change, with the tool a switch moves from, for the sentences.
    change: RecordChange,
    /// The folder the definition named, for the sentences.
    old_folder: String,
    /// The value at the cursor, which the change is made on.
    edited: EditedReconstruction,
}

/// Everything the build needs, owned.
struct SiftFilesPlan {
    /// The node's label, for the sentences.
    label: String,
    /// Each image's photograph, in the node's order.
    images: Vec<PathBuf>,
    options: BuildSiftFilesOptions,
    switch: Option<Switch>,
}

/// The build, then the change of the definition when there is one.
///
/// The change is pushed only when every file was built, so a build that is
/// cancelled or fails leaves the definition as it was. The files it wrote stay,
/// and a later build skips them. The sentence names the folder written to and,
/// after a change, the folder the definition moved from.
fn build(plan: SiftFilesPlan, progress: &Progress<'_>) -> Finished {
    let SiftFilesPlan {
        label,
        images,
        options,
        switch,
    } = plan;
    let report = match build_sift_files(&images, &options, progress) {
        Ok(report) => report,
        Err(BuildSiftFilesError::Cancelled) => return Finished::Cancelled,
        Err(e) => {
            return Finished::Failed(format!("Could not build the .sift files of {label}: {e}"))
        }
    };
    let built = format!(
        "{} written and {} already up to date, in {}",
        report.extracted,
        report.skipped,
        options.feature_prefix_dir.display()
    );
    let Some(Switch {
        change,
        old_folder,
        edited,
    }) = switch
    else {
        // The files are the workspace's and not the version's: nothing about
        // the reconstruction moved, so the row is the whole of it.
        return Finished::NoChange(format!("Built the .sift files of {label}: {built}"));
    };
    // The definition is the base's, so a value with point edits over its base
    // is materialised first, as every whole-value edit does.
    let (mut value, map) = if edited.deleted_points.is_empty() && edited.added.points.is_empty() {
        (edited.base.clone_for_edit(), PointMap::Removed(Vec::new()))
    } else {
        let _phase = progress.phase("materialise");
        let (value, map) = edited.materialize();
        (value, PointMap::Rows(map))
    };
    // The options the files were built with, so the node looks for them where
    // they are. The paths of the definition are left alone.
    let contents = &mut value.metadata.workspace.contents;
    contents.feature_tool = SFMTOOL_FEATURE_TOOL.to_string();
    contents.feature_type = SFMTOOL_FEATURE_TYPE.to_string();
    contents.feature_options = options.feature_options;
    contents.feature_prefix_dir = options.feature_prefix_dir.to_string_lossy().into_owned();
    let moved = format!("moved from {old_folder}");
    let (version_label, text) = match change {
        RecordChange::Switch(from) => {
            let version_label = format!("Switched the SIFT extraction of {label} to sfmtool");
            let from = match from.is_empty() {
                true => "no tool".to_string(),
                false => format!("'{from}'"),
            };
            let text =
                format!("{version_label} from {from}, and built its .sift files: {built}, {moved}");
            (version_label, text)
        }
        _ => {
            let version_label =
                format!("Converted the SIFT extraction record of {label} to version 2");
            let text = format!("{version_label}, and built its .sift files: {built}, {moved}");
            (version_label, text)
        }
    };
    Finished::Produced {
        value,
        map,
        text,
        version_label,
    }
}
