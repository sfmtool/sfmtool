// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Opening a file: a background task that reads the `.sfmr`, fills in what it
//! does not carry for display, and hands the GUI thread a node to append.
//!
//! See `specs/gui/background-tasks.md` section "Opening a file". File > Open,
//! a path on the command line and the MCP `open_reconstruction` tool all start
//! the same [`Operation::OPEN`] task. For each file it runs, on the worker:
//!
//! - **the read**, `SfmrReconstruction::load`, whose `read`, `convert
//!   convention` and `derive` stages nest under the file's `open` phase;
//! - **`thumbnails`**, for a file that carries none: every row built at once
//!   ([`DisplayThumbnails::build`]), from the image's verified `.sift` first
//!   and its photograph second, held by the node and never by the value;
//! - **`patch bitmaps`**, for a file with patch frames and inline keypoints but
//!   no bitmaps: every photograph decoded, then every patch fused at its stored
//!   frame and keypoints by the fuse `sfm xform --add-patch-bitmaps` runs
//!   ([`fuse_patch_cloud_bitmaps`]). The column goes into the value marked
//!   [`sfmtool_core::PointSet::patch_bitmaps_for_display`], so the bench, the
//!   edits and Track View read it as they would a file's own, while no save
//!   writes it and no content hash covers it.
//!
//! The GUI thread then appends one node per file, in the order asked for
//! ([`AppState::append_opened`]).

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use rayon::prelude::*;
use sfmtool_core::camera::remap::ImageU8Pyramid;
use sfmtool_core::geometry::RigidTransform;
use sfmtool_core::patch::keypoint_subpixel::{fuse_patch_cloud_bitmaps, KeypointSubpixelParams};
use sfmtool_core::patch::normal_refine::ProjectedImage;
use sfmtool_core::patch::PatchCloud;
use sfmtool_core::progress::{Cancelled, Progress};
use sfmtool_core::progress_note;
use sfmtool_core::SfmrReconstruction;

use crate::action_log::{Actor, Kind};
use crate::background::{Finished, Job, Operation};
use crate::display_thumbnails::DisplayThumbnails;
use crate::scene::{ReconId, SceneNode};

use super::{AppState, PYRAMID_LEVELS};

#[cfg(test)]
pub(crate) mod tests;

/// One file an open was asked for, and what became of it.
pub(crate) struct OpenedFile {
    /// The path as it was asked for.
    pub(crate) path: PathBuf,
    /// The value read and filled in, or the sentence saying why it was not.
    pub(crate) outcome: Result<Loaded, String>,
}

/// A file read and made ready to show.
pub(crate) struct Loaded {
    /// The value, carrying the file's own content hash. Its patch bitmap
    /// column, when the open rendered one, is marked for display.
    pub(crate) recon: SfmrReconstruction,
    /// The thumbnails the open built for a file that carries none. `None` for
    /// a file with its own, whose node shares that column, and for one where
    /// neither a `.sift` nor a photograph could supply a single row.
    pub(crate) display_thumbnails: Option<Arc<DisplayThumbnails>>,
}

/// Why a file's open stopped short of a value.
enum Stop {
    /// The task was asked to stop.
    Cancelled,
    /// The file could not be read, in the words the Action Log shows.
    Failed(String),
}

impl From<Cancelled> for Stop {
    fn from(_: Cancelled) -> Self {
        Stop::Cancelled
    }
}

impl AppState {
    /// Start opening `paths`, in order, as one background task.
    ///
    /// Refused without starting anything when an operation is already running
    /// (one at a time, viewer-wide) or when a path is not a file, which is the
    /// one failure knowable before the read. **Nothing is logged here**: the
    /// caller phrases a refusal in its own vocabulary, and the task's own row
    /// is written when it lands.
    pub fn start_open(&mut self, paths: Vec<PathBuf>) -> Result<(), String> {
        if paths.is_empty() {
            return Err("There is no file to open.".to_string());
        }
        if let Some(why) = self.running_refusal() {
            return Err(why);
        }
        if let Some(missing) = paths.iter().find_map(|path| not_a_file(path)) {
            return Err(missing);
        }
        let label = match paths.as_slice() {
            [path] => crate::scene::label_for_path(path),
            many => format!("{} files", many.len()),
        };
        self.start_task(Operation::OPEN, None, label, open_job(paths))
    }

    /// Open `paths` for the File menu and the command line: every path that is
    /// not a file is a failed row of its own, and the rest open as one task.
    ///
    /// The difference from [`Self::start_open`] is only who writes the
    /// refusals. A person who picked five files, one of them since deleted,
    /// asked for the other four as well.
    pub fn open_files(&mut self, paths: Vec<PathBuf>) {
        let (found, missing): (Vec<PathBuf>, Vec<PathBuf>) =
            paths.into_iter().partition(|path| path.is_file());
        for path in &missing {
            if let Some(message) = not_a_file(path) {
                self.action_log.fail(Kind::File, message);
            }
        }
        if found.is_empty() {
            return;
        }
        if let Err(message) = self.start_open(found) {
            self.action_log.fail(Kind::File, message);
        }
    }

    /// Append each file an open read as a node, in order, and write a failed
    /// row, as `actor`, for each it could not.
    ///
    /// Returns the nodes made and, when no file was read, the last refusal,
    /// which the task ends with: a one-file open that failed is then one failed
    /// row rather than two.
    pub(crate) fn append_opened(
        &mut self,
        files: Vec<OpenedFile>,
        actor: Actor,
    ) -> (Vec<ReconId>, Option<String>) {
        let any_read = files.iter().any(|file| file.outcome.is_ok());
        let mut made = Vec::new();
        let mut refusals = Vec::new();
        for OpenedFile { path, outcome } in files {
            match outcome {
                Ok(Loaded {
                    recon,
                    display_thumbnails,
                }) => {
                    log::info!(
                        "Loaded {} points, {} images from {}",
                        recon.point_count(),
                        recon.image_count(),
                        path.display()
                    );
                    let mut node = SceneNode::from_path(&path, recon);
                    if display_thumbnails.is_some() {
                        node.display_thumbnails = display_thumbnails;
                    }
                    made.push(self.append_node(node));
                }
                Err(message) => {
                    log::error!("{message}");
                    refusals.push(message);
                }
            }
        }
        let last_refusal = if any_read { None } else { refusals.pop() };
        let standing = self.action_log.actor();
        self.action_log.set_actor(actor);
        for message in refusals {
            self.action_log.fail(Kind::File, message);
        }
        self.action_log.set_actor(standing);
        (made, last_refusal)
    }
}

/// The Action Log sentence of an open that made `made`: each node's label, as
/// the tree shows it once deduplicated, and the path it came from.
pub(crate) fn opened_sentence(state: &AppState, made: &[ReconId]) -> String {
    let parts: Vec<String> = made
        .iter()
        .filter_map(|&id| state.node(id))
        .map(|node| {
            let path = node
                .path
                .as_ref()
                .map(|path| path.display().to_string())
                .unwrap_or_default();
            format!("{} from {path}", node.label)
        })
        .collect();
    format!("Opened {}", parts.join("; "))
}

/// The refusal for a path that is not a file, or `None` when it is one.
fn not_a_file(path: &Path) -> Option<String> {
    match std::fs::metadata(path) {
        Ok(meta) if meta.is_file() => None,
        Ok(_) => Some(format!(
            "Failed to load {}: it is not a file",
            path.display()
        )),
        Err(e) => Some(format!("Failed to load {}: {e}", path.display())),
    }
}

/// The work of one open: each path read and filled in, in order, each under an
/// equal share of the bar.
pub(crate) fn open_job(paths: Vec<PathBuf>) -> Job {
    Box::new(move |progress| {
        let n = paths.len();
        let mut files = Vec::with_capacity(n);
        for (path, share) in paths.into_iter().zip(progress.split_evenly(n)) {
            if progress.is_cancelled() {
                return Finished::Cancelled;
            }
            let mut phase = share.phase("open");
            if n > 1 {
                progress_note!(phase, "{}", path.display());
            }
            match load_for_display(&path, &phase) {
                Ok(loaded) => files.push(OpenedFile {
                    path,
                    outcome: Ok(loaded),
                }),
                Err(Stop::Cancelled) => return Finished::Cancelled,
                Err(Stop::Failed(message)) => files.push(OpenedFile {
                    path,
                    outcome: Err(message),
                }),
            }
        }
        Finished::Opened(files)
    })
}

/// Read `path` and fill in what it does not carry for display.
fn load_for_display(path: &Path, progress: &Progress<'_>) -> Result<Loaded, Stop> {
    let [read, rest] = progress.split([1.0, 9.0]);
    let mut recon = SfmrReconstruction::load(path, &read)
        .map_err(|e| Stop::Failed(format!("Failed to load {}: {e}", path.display())))?;
    read.set_fraction(1.0);
    progress.check_cancel()?;

    let wants_thumbnails = recon.image_table.thumbnails_y_x_rgb.is_none();
    let wants_bitmaps = recon.point_set.patch_u_halfvec_xyz.is_some()
        && recon.point_set.patch_bitmaps_y_x_rgba.is_none()
        && recon.keypoints_xy().is_some();
    // Weighed by what each costs: a thumbnail is mostly a `.sift` read, and a
    // bitmap column is a decode of every photograph and a fuse per point.
    let [thumbnails, bitmaps] = rest.split([
        if wants_thumbnails { 1.0 } else { 0.0 },
        if wants_bitmaps { 8.0 } else { 0.0 },
    ]);

    let mut display_thumbnails = None;
    if wants_thumbnails {
        let mut phase = thumbnails.phase("thumbnails");
        let (display, from) = DisplayThumbnails::build(&recon, &phase)?;
        progress_note!(
            phase,
            "{} from .sift files, {} from photographs, {} placeholders",
            from.sift,
            from.photographs,
            from.placeholders
        );
        display_thumbnails = display;
    }
    if wants_bitmaps {
        let phase = bitmaps.phase("patch bitmaps");
        if let Some(column) = render_patch_bitmaps(&recon, &phase)? {
            recon.point_set.patch_bitmaps_y_x_rgba = Some(Arc::new(column));
            recon.point_set.patch_bitmaps_for_display = true;
        }
    }
    progress.set_fraction(1.0);
    Ok(Loaded {
        recon,
        display_thumbnails,
    })
}

/// Render `recon`'s patch bitmap column at its stored frames and keypoints,
/// moving nothing.
///
/// Two stages under `progress`: `decode photographs`, each image's photograph
/// read and pyramided in parallel, and `fuse`, the whole-cloud form of the one
/// fuse the bench commit and `--add-patch-bitmaps` use. A photograph that
/// cannot be read, or is not the size its camera says, is left out of every
/// patch's views rather than failing the open; a point that two readable views
/// do not see gets a zero row. `Ok(None)` when not one photograph could be
/// read, since a column of zero rows would draw nothing.
fn render_patch_bitmaps(
    recon: &SfmrReconstruction,
    progress: &Progress<'_>,
) -> Result<Option<ndarray::Array4<u8>>, Cancelled> {
    let Some(cloud) = PatchCloud::from_stored_frames(recon) else {
        return Ok(None);
    };
    let [decode, fuse] = progress.split([1.0, 3.0]);
    let images = &recon.image_table.images;
    let total = images.len();
    let pyramids: Vec<Option<ImageU8Pyramid>> = {
        let mut phase = decode.phase("decode photographs");
        let landed = AtomicUsize::new(0);
        let pyramids: Vec<Option<ImageU8Pyramid>> = images
            .par_iter()
            .map(|image| {
                if phase.is_cancelled() {
                    return None;
                }
                let camera = &recon.image_table.cameras[image.camera_index as usize];
                let decoded = crate::state::decode_full_res(&recon.workspace_dir.join(&image.name))
                    .filter(|decoded| {
                        decoded.width() == camera.width && decoded.height() == camera.height
                    })
                    .map(|decoded| ImageU8Pyramid::from_image(decoded, PYRAMID_LEVELS));
                let n = landed.fetch_add(1, Ordering::Relaxed) + 1;
                phase.count(n as u64, Some(total as u64), "images");
                decoded
            })
            .collect();
        phase.check_cancel()?;
        let read = pyramids.iter().filter(|p| p.is_some()).count();
        progress_note!(phase, "{read} of {total} read");
        pyramids
    };
    if pyramids.iter().all(Option::is_none) {
        return Ok(None);
    }
    let poses: Vec<RigidTransform> = images
        .iter()
        .map(|image| {
            let q = image.quaternion_wxyz;
            RigidTransform::from_wxyz_translation(
                [q.w, q.i, q.j, q.k],
                [
                    image.translation_xyz.x,
                    image.translation_xyz.y,
                    image.translation_xyz.z,
                ],
            )
        })
        .collect();
    let views: Vec<Option<ProjectedImage<'_>>> = images
        .iter()
        .zip(&poses)
        .zip(&pyramids)
        .map(|((image, cam_from_world), pyramid)| {
            pyramid.as_ref().map(|pyramid| ProjectedImage {
                camera: &recon.image_table.cameras[image.camera_index as usize],
                cam_from_world,
                pyramid,
            })
        })
        .collect();
    let mut phase = fuse.phase("fuse");
    let column = fuse_patch_cloud_bitmaps(
        &cloud,
        recon,
        &views,
        &KeypointSubpixelParams::default(),
        None,
        &phase,
    )?;
    progress_note!(phase, "{} patches at {} px", cloud.len(), column.shape()[1]);
    Ok(Some(column))
}

#[cfg(test)]
impl AppState {
    /// Open `path` and drive the task to its end, the way the frames would:
    /// the node it made, or the refusal it ended with.
    pub(crate) fn open_now(&mut self, path: &Path) -> Result<ReconId, String> {
        self.start_open(vec![path.to_path_buf()])?;
        self.finish_background_task();
        let last = self
            .last_background_task
            .as_ref()
            .expect("the open just finished");
        match &last.outcome {
            Ok(_) => last
                .opened
                .ok_or_else(|| "the open made no node".to_string()),
            Err(message) => Err(message.clone()),
        }
    }
}
