// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! *Create Track Here*: a track built at one pixel of a posed photograph, put
//! on the node's bench and committed, as one gesture.
//!
//! See `specs/gui/bench.md` section "Create Track Here". The track is core's
//! `sfmtool_core::bench::build_track_at_pixel` (`specs/core/bench/track-at-pixel.md`),
//! run on a worker through the background machinery. What this module adds is
//! what the viewer owns around it:
//!
//! - **the sources**: the node's index files, read only when they are
//!   `current`, so a member whose file is missing or stale refuses and names
//!   what it lacked;
//! - **the landing**: the track put on the bench as the active item (one bench
//!   version), and then committed by [`AppState::commit_bench_track`], the step
//!   Track View's *Commit* button takes (one edit version), so an Undo takes
//!   back the point first and leaves the track on the bench;
//! - **the refusal**: one failed Action Log row in one sentence, the last
//!   member's stage and reason, with every member's refusal among the row's
//!   detail lines and the index files named when they were missing or stale.

use std::path::PathBuf;
use std::sync::Arc;

use sfmtool_core::bench::{
    build_track_at_pixel, BenchItem, ClusterSeed, EditableTrack, MatchesClusters, MemberRefusal,
    SiftIndexSource, StageRecord, TrackAtPixelError, TrackAtPixelOptions, TrackAtPixelReport,
    TrackAtPixelSources,
};
use sfmtool_core::features::kdforest::{ImageKeypoints, LazyKdForestU8};
use sfmtool_core::progress::Progress;

use crate::action_log::{version_step_text, Actor, Kind};
use crate::background::{Finished, Job, Operation};
use crate::index_files::IndexFileState;
use crate::scene::{ImageRef, ReconId};
use crate::state::AppState;

use super::Committed;

#[cfg(test)]
pub(crate) mod tests;

/// The Image Detail context-menu entry's label, which the tests aim at.
pub(crate) const CREATE_TRACK_HERE_LABEL: &str = "Create Track Here";

/// The pointer gesture that does the same at the clicked pixel, as the menu
/// entry shows it beside its label: the primary button with Control and Shift
/// held.
pub(crate) const CREATE_TRACK_HERE_SHORTCUT: &str = "Ctrl+Shift+Click";

/// Whether `modifiers` are the ones [`CREATE_TRACK_HERE_SHORTCUT`] names.
///
/// The physical Control key on every platform, `ctrl` rather than egui's
/// `command`, because the gesture is named for that key; Alt must be up, so a
/// chord with a third modifier is not read as this one.
pub(crate) fn is_create_track_chord(modifiers: egui::Modifiers) -> bool {
    modifiers.ctrl && modifiers.shift && !modifiers.alt
}

/// Whether `image` of `recon` carries a pose: every component of its rotation
/// and its translation finite, the rule the Scene tree's *Resect Image* greys
/// by. A non-finite one is a placeholder rather than a registration.
fn is_posed(recon: &sfmtool_core::SfmrReconstruction, image: usize) -> bool {
    recon.image_table.images.get(image).is_some_and(|row| {
        row.quaternion_wxyz.coords.iter().all(|c| c.is_finite())
            && row.translation_xyz.iter().all(|c| c.is_finite())
    })
}

/// Why *Create Track Here* is greyed over an image with no pose.
pub(crate) const NOT_POSED: &str =
    "This image is not posed, so there is no ray through the pixel to build a track along.";

/// Why *Create Track Here* is greyed on a node whose observations are `.sift`
/// feature indexes: the commit it ends in writes keypoints inline, which only
/// an `embedded_patches` reconstruction stores.
pub(crate) const NOT_EMBEDDED_PATCHES: &str =
    "Creating a track commits it, and committing needs an embedded_patches reconstruction; \
     this one's observations are .sift features. Convert it to embedded patches first.";

/// One run's inputs and answer, carried home from the worker.
pub(crate) struct TrackAtPixelRun {
    /// The queried image, by its index in the node.
    pub(crate) image: u32,
    /// The queried pixel, in that image's own pixels.
    pub(crate) pixel: [f64; 2],
    /// The image's name, for the sentences.
    pub(crate) image_name: String,
    /// The image's name without its extension, which the label is minted
    /// from.
    pub(crate) image_stem: String,
    /// What the node's index files lacked when the run started, or `None` when
    /// both were current.
    pub(crate) index_note: Option<String>,
    /// The track and how it was built, or every member's refusal in order.
    pub(crate) result: Result<(EditableTrack, TrackAtPixelReport), Vec<MemberRefusal>>,
}

/// What one *Create Track Here* left behind, which the wire answers with.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum CreatedTrack {
    /// Built, put on the bench as `item`, and committed.
    Committed {
        /// The bench item the track is, now seated on the point.
        item: String,
        /// The cascade member that built it.
        member: &'static str,
        /// The point the commit wrote.
        point: Committed,
    },
    /// Built and put on the bench as `item`, and the commit refused.
    NotCommitted {
        /// The bench item the track is.
        item: String,
        /// The cascade member that built it.
        member: &'static str,
        /// The failed row's sentence.
        why: String,
    },
    /// No member built a track.
    Refused {
        /// Each member's refusal, one line each, in the order they were tried.
        refusals: Vec<String>,
    },
}

/// One member's refusal as the detail line and the wire spell it.
fn refusal_line(refusal: &MemberRefusal) -> String {
    format!(
        "{} refused at {}: {}",
        refusal.member, refusal.stage, refusal.reason
    )
}

/// The in count and the median ZNCC the final gates judged, when the member
/// got that far.
fn final_reading(stages: &[StageRecord]) -> Option<(usize, f64)> {
    stages.iter().rev().find_map(|stage| match stage {
        StageRecord::Final {
            in_views,
            zncc_median,
            ..
        } => Some((*in_views, *zncc_median)),
        _ => None,
    })
}

impl AppState {
    /// Why *Create Track Here* cannot run on `image`, or `None` when it can.
    ///
    /// What greys the Image Detail entry with its hover text and what the step
    /// and the wire refuse with, so the three say one thing. The index files
    /// are not asked about: two of the cascade's four members read neither
    /// file, so a node without them can still have a track built, and a run
    /// they would have helped says so in its own refusal.
    pub(crate) fn create_track_here_refusal(&self, image: ImageRef) -> Option<String> {
        if let Some(why) = self.busy_refusal(image.recon) {
            return Some(why);
        }
        let Some(node) = self.node(image.recon) else {
            return Some("That reconstruction is no longer loaded.".to_string());
        };
        let recon = node.recon();
        if image.index() >= recon.image_table.images.len() {
            return Some(format!(
                "{} has {} images; there is no image {}.",
                node.label,
                recon.image_table.images.len(),
                image.index()
            ));
        }
        if !is_posed(recon, image.index()) {
            return Some(NOT_POSED.to_string());
        }
        if node.edited().has_feature_indexes() {
            return Some(NOT_EMBEDDED_PATCHES.to_string());
        }
        None
    }

    /// *Create Track Here* at `pixel` of `image`, from the Image Detail menu
    /// entry or its Control+Shift click: the job started, or one failed row in
    /// the refusal's words.
    pub(crate) fn create_track_here(&mut self, image: ImageRef, pixel: [f32; 2]) {
        let pixel = [f64::from(pixel[0]), f64::from(pixel[1])];
        // The refusal is already the log row `start_create_track_at_pixel`
        // writes, so there is nothing more to say here.
        let _ = self.start_create_track_at_pixel(image, pixel);
    }

    /// Build a track at `pixel` of `image` on a worker, then put it on the
    /// node's bench and commit it when the worker comes home.
    ///
    /// What returns here is whether the run could **begin**. A refusal in
    /// front of the worker is one failed `Bench` row and no task; the run's own
    /// refusal, which is the cascade's, lands with the task.
    pub(crate) fn start_create_track_at_pixel(
        &mut self,
        image: ImageRef,
        pixel: [f64; 2],
    ) -> Result<(), String> {
        let outcome = self
            .create_track_at_pixel_job(image, pixel)
            .and_then(|job| {
                self.start_background_task(Operation::CREATE_TRACK_AT_PIXEL, image.recon, job)
            });
        if let Err(message) = &outcome {
            self.action_log.fail(Kind::Bench, message.clone());
        }
        outcome
    }

    /// The run as a closure that owns everything it reads.
    ///
    /// Crate-visible for the reason [`AppState::bench_evaluate_job`] is: the
    /// test that holds the operation's cancellable declaration to its claim
    /// runs the real work.
    ///
    /// **Only a `current` index file is handed over.** A file that is missing
    /// or stale is left out, so the member that reads it refuses and names what
    /// it lacked, and the note saying which files those were is taken now, as
    /// the run starts, for the refusal's sentence.
    pub(crate) fn create_track_at_pixel_job(
        &mut self,
        image: ImageRef,
        pixel: [f64; 2],
    ) -> Result<Job, String> {
        if let Some(why) = self.create_track_here_refusal(image) {
            return Err(why);
        }
        let id = image.recon;
        let image_name = self.image_name(image);
        let camera = self
            .image_camera(image)
            .ok_or_else(|| format!("{image_name} has no camera."))?;
        let (clamped, _) = sfmtool_core::bench::clamp_to_photograph(&camera, pixel);
        if clamped.is_some() {
            return Err(format!(
                "Cannot create a track at ({:.1}, {:.1}): that is not on the {}x{} photograph \
                 {image_name}.",
                pixel[0], pixel[1], camera.width, camera.height
            ));
        }
        // The files are opened on sight when they are there and nothing has
        // looked yet, and re-judged when a version has moved the image table,
        // so the states read below are the node's as it stands.
        self.refresh_index_files(id);
        let index_note = self.index_files_note(id);
        let forest = (self.sift_index_state(id) == IndexFileState::Current)
            .then(|| self.sift_index(id).map(|index| Arc::clone(&index.forest)))
            .flatten();
        let cluster_patches = (self.cluster_patches_state(id) == IndexFileState::Current)
            .then(|| self.cluster_patches(id).map(|file| file.path.clone()))
            .flatten();

        let node = self
            .node(id)
            .ok_or_else(|| "That reconstruction is no longer loaded.".to_string())?;
        let recon = node.recon();
        let image_names: Vec<String> = recon
            .image_table
            .images
            .iter()
            .map(|row| row.name.clone())
            .collect();
        // Every image's `.sift` file, for the keypoints the constellation
        // member queries the index with; read on the worker, and only when
        // there is an index to query.
        let sift_files: Vec<PathBuf> = match forest {
            Some(_) => (0..image_names.len())
                .map(|i| recon.sift_path_for_image(i))
                .collect(),
            None => Vec::new(),
        };
        let edited = node.edited().clone();
        let image_stem = std::path::Path::new(&image_name)
            .file_stem()
            .map_or_else(|| image_name.clone(), |s| s.to_string_lossy().into_owned());
        let every: Vec<usize> = (0..image_names.len()).collect();
        let views = self.view_sources_for(id, &every)?;
        let plan = Plan {
            image: image.index() as u32,
            pixel,
            image_name,
            image_stem,
            index_note,
            forest,
            sift_files,
            cluster_patches,
            image_names,
        };
        Ok(Box::new(move |progress| {
            if progress.is_cancelled() {
                return Finished::Cancelled;
            }
            let decoded = match views.decode(progress) {
                Ok(decoded) => decoded,
                Err(e) => {
                    return Finished::Failed(format!(
                        "Cannot create a track in {}: {e}",
                        plan.image_name
                    ))
                }
            };
            if progress.is_cancelled() {
                return Finished::Cancelled;
            }
            run(plan, &edited, &decoded.views(), progress)
        }))
    }

    /// What the node's index files lack, as the clause a refusal carries, or
    /// `None` when both are current.
    ///
    /// Each file is named with the member that reads it, because that is why a
    /// person reading the refusal should care: the constellation member
    /// queries the SIFT index and the clusters member reads the cluster
    /// patches. It ends by naming *Build Index Files*, or by saying why the
    /// node cannot have them yet.
    pub(crate) fn index_files_note(&self, id: ReconId) -> Option<String> {
        let describe = |state: IndexFileState, file: &str, member: &str| match state {
            IndexFileState::Current => None,
            IndexFileState::None => Some(format!(
                "No {file} is open, so the {member} member had nothing to read."
            )),
            IndexFileState::Stale => Some(format!(
                "The {file} is out of date, so the {member} member did not read it."
            )),
        };
        let mut sentences: Vec<String> = [
            describe(self.sift_index_state(id), "SIFT index", "constellation"),
            describe(
                self.cluster_patches_state(id),
                "cluster patches file",
                "clusters",
            ),
        ]
        .into_iter()
        .flatten()
        .collect();
        if sentences.is_empty() {
            return None;
        }
        sentences.push(match self.index_files_home_refusal(id) {
            Some(why) => why,
            None => format!(
                "{} (the Index Files row in the Scene tree) makes both.",
                crate::index_files::BUILD_INDEX_FILES
            ),
        });
        Some(sentences.join(" "))
    }

    /// Land a finished run on the node at `index`: the track put on the bench
    /// as the active item, one bench version, or the cascade's refusal.
    ///
    /// Gives back the row's outcome, what the wire will answer with, and, for
    /// a track that landed, the item to commit once the row is written: the
    /// commit is its own version and its own row, and it comes after this one
    /// because that is the order the two happened in.
    pub(crate) fn land_track_at_pixel(
        &mut self,
        index: usize,
        run: TrackAtPixelRun,
    ) -> (Result<String, String>, CreatedTrack, Option<String>) {
        let TrackAtPixelRun {
            image,
            pixel,
            image_name,
            image_stem,
            index_note,
            result,
        } = run;
        let at = format!("({:.1}, {:.1}) in {image_name}", pixel[0], pixel[1]);
        match result {
            Err(refusals) => {
                let last = refusals.last();
                let mut sentence = match last {
                    Some(last) => format!(
                        "Cannot create a track at {at}: every member refused; the last, {}, \
                         at {}: {}",
                        last.member, last.stage, last.reason
                    ),
                    None => format!("Cannot create a track at {at}: no member was asked"),
                };
                if let Some(note) = index_note {
                    sentence.push_str(". ");
                    sentence.push_str(&note);
                }
                let refusals = refusals.iter().map(refusal_line).collect();
                (Err(sentence), CreatedTrack::Refused { refusals }, None)
            }
            Ok((track, report)) => {
                let member = report.member.name();
                // The label a cluster started at the same pixel would take,
                // so a person who knows one knows the other.
                let base = ClusterSeed::from_pixel(image, image_stem, pixel, 1.0).label();
                let bench = Arc::clone(self.scene[index].history.current_bench());
                let (next, item) = bench.put(&base, BenchItem::Track(Arc::new(track)));
                let version_label = format!("Created {item} at {at} with the {member} member");
                let mut text = version_label.clone();
                if let Some((in_views, zncc)) = final_reading(&report.stages) {
                    text.push_str(&format!(
                        ": {in_views} observations in, median ZNCC {zncc:.2}"
                    ));
                }
                if !report.refusals.is_empty() {
                    let before: Vec<&str> =
                        report.refusals.iter().map(|r| r.member.name()).collect();
                    text.push_str(&format!(", after {} refused", before.join(", ")));
                }
                let node = &mut self.scene[index];
                let serial = node.history.push_bench(Arc::new(next), version_label);
                let parent = crate::state::edits::version_before(node, serial);
                let placeholder = CreatedTrack::NotCommitted {
                    item: item.clone(),
                    member,
                    why: String::new(),
                };
                (
                    Ok(version_step_text(&text, parent, serial)),
                    placeholder,
                    Some(item),
                )
            }
        }
    }

    /// Commit the item a run just put on `id`'s bench, as `actor`, and say
    /// what became of it.
    ///
    /// Through [`AppState::commit_bench_track`], the step Track View's
    /// *Commit* takes, so the version, its `Edit` row, the selection of the
    /// written point and the item left seated on it are that step's own. A
    /// refusal is one failed `Edit` row and leaves the track on the bench.
    pub(crate) fn commit_created_track(
        &mut self,
        actor: Actor,
        id: ReconId,
        item: String,
        member: &'static str,
    ) -> CreatedTrack {
        let standing = self.action_log.actor();
        self.action_log.set_actor(actor);
        let created = match self.commit_bench_track(id, &item) {
            Ok(point) => CreatedTrack::Committed {
                item,
                member,
                point,
            },
            Err(why) => {
                let why = format!("{why}; it stays on the bench as {item}.");
                self.action_log.fail(Kind::Edit, why.clone());
                CreatedTrack::NotCommitted { item, member, why }
            }
        };
        self.action_log.set_actor(standing);
        created
    }
}

/// Everything the worker reads beside the views and the value, owned.
struct Plan {
    image: u32,
    pixel: [f64; 2],
    image_name: String,
    image_stem: String,
    index_note: Option<String>,
    /// The SIFT index, when it is current.
    forest: Option<Arc<LazyKdForestU8>>,
    /// Every image's `.sift` path, in the node's order, when there is an index.
    sift_files: Vec<PathBuf>,
    /// The cluster-patches file, when it is current.
    cluster_patches: Option<PathBuf>,
    /// Every image's name, in the node's order, which the clusters are
    /// indexed onto.
    image_names: Vec<String>,
}

/// The worker's half after the decode: read what the members need, run the
/// cascade, and hand back the track or every refusal.
fn run(
    plan: Plan,
    edited: &sfmtool_core::EditedReconstruction,
    views: &[sfmtool_core::patch::normal_refine::ProjectedImage<'_>],
    progress: &Progress<'_>,
) -> Finished {
    let keypoints: Option<Vec<ImageKeypoints>> = plan.forest.as_ref().map(|_| {
        let _phase = progress.phase("read keypoints");
        plan.sift_files
            .iter()
            .map(|path| {
                ImageKeypoints::read(path).unwrap_or_else(|e| {
                    sfmtool_core::progress_warn!(
                        progress,
                        "{e}; that image has no keypoints to query with"
                    );
                    ImageKeypoints {
                        positions: Vec::new(),
                        affine_shapes: Vec::new(),
                    }
                })
            })
            .collect()
    });
    if progress.is_cancelled() {
        return Finished::Cancelled;
    }
    let clusters: Option<MatchesClusters> = plan.cluster_patches.as_ref().and_then(|path| {
        let _phase = progress.phase("read cluster patches");
        let names: Vec<&str> = plan.image_names.iter().map(String::as_str).collect();
        let read = sfmtool_matches_format::read_matches(path)
            .map_err(|e| e.to_string())
            .and_then(|data| MatchesClusters::new(&data, &names).map_err(|e| e.to_string()));
        match read {
            Ok(clusters) => Some(clusters),
            Err(e) => {
                sfmtool_core::progress_warn!(
                    progress,
                    "Cannot read the cluster patches {}: {e}",
                    path.display()
                );
                None
            }
        }
    });
    if progress.is_cancelled() {
        return Finished::Cancelled;
    }
    let sources = TrackAtPixelSources {
        sift_index: match (&plan.forest, &keypoints) {
            (Some(forest), Some(keypoints)) => Some(SiftIndexSource { forest, keypoints }),
            _ => None,
        },
        clusters: clusters.as_ref(),
    };
    let result = build_track_at_pixel(
        edited,
        views,
        &sources,
        plan.image,
        plan.pixel,
        &TrackAtPixelOptions::default(),
        progress,
    );
    let result = match result {
        Ok(built) => Ok(built),
        Err(TrackAtPixelError::Cancelled) => return Finished::Cancelled,
        Err(TrackAtPixelError::Refused { refusals }) => {
            // Every member's refusal among the row's detail lines, which is
            // where a person expanding the failed row reads them.
            for refusal in &refusals {
                sfmtool_core::progress_warn!(progress, "{}", refusal_line(refusal));
            }
            Err(refusals)
        }
        Err(e) => {
            return Finished::Failed(format!(
                "Cannot create a track at ({:.1}, {:.1}) in {}: {e}",
                plan.pixel[0], plan.pixel[1], plan.image_name
            ))
        }
    };
    Finished::TrackAtPixel(Box::new(TrackAtPixelRun {
        image: plan.image,
        pixel: plan.pixel,
        image_name: plan.image_name,
        image_stem: plan.image_stem,
        index_note: plan.index_note,
        result,
    }))
}
