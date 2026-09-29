// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! *Find Nearby Tracks*: the tracks the photographs agree on near one pixel of
//! a posed photograph, put on the node's bench and committed, as one version.
//!
//! See `specs/gui/bench.md` section "Find Nearby Tracks". The search is core's
//! `sfmtool_core::bench::find_nearby_tracks`
//! (`specs/core/bench/nearby-tracks.md`), run on a worker through the
//! background machinery. What this module adds is what the viewer owns around
//! it:
//!
//! - **the sources**: the node's index files, read only when they are
//!   `current`, as *Create Track Here* reads them, and every image's `.sift`
//!   file for guided matching when every one of them is on disk;
//! - **the landing**: every usable track put on the bench under its label, the
//!   existing points' own tracks as *Edit on Bench* puts them, the rest
//!   committed as new points, all in **one** version, so one undo takes the
//!   whole find back;
//! - **the empty answer**: a query that finds nothing usable writes one row
//!   saying so and pushes no version.

use std::path::PathBuf;
use std::sync::Arc;

use rayon::prelude::*;
use sfmtool_core::bench::{
    self, find_nearby_tracks, BenchItem, CreateTrackOptions, GreyImages, GuidedSource,
    ImageDescriptors, KeypointRays, MatchesClusters, NearbyTrackOptions, NearbyTrackSources,
    NearbyTracks, NearbyTracksError, SiftIndexSource,
};
use sfmtool_core::features::kdforest::{ImageKeypoints, LazyKdForestU8};
use sfmtool_core::progress::Progress;
use sfmtool_core::EditedReconstruction;

use crate::action_log::{version_step_text, Kind};
use crate::background::{Finished, Job, Operation};
use crate::document::{CreatedPoints, PointMap};
use crate::index_files::IndexFileState;
use crate::scene::{ImageRef, PointRef};
use crate::state::AppState;

#[cfg(test)]
pub(crate) mod tests;

/// The Image Detail context-menu entry's label, which the tests aim at.
pub(crate) const FIND_NEARBY_TRACKS_LABEL: &str = "Find Nearby Tracks";

/// One run's inputs and answer, carried home from the worker.
pub(crate) struct NearbyTracksRun {
    /// The queried pixel, in the queried image's own pixels.
    pub(crate) pixel: [f64; 2],
    /// The queried image's name, for the sentences.
    pub(crate) image_name: String,
    /// Whether the tracks that are not existing points are committed.
    pub(crate) commit: bool,
    /// What core found.
    pub(crate) found: NearbyTracks,
}

/// What became of one usable track when a run landed.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Landing {
    /// An existing point: its own track is on the bench as `item`, seated on
    /// it, and nothing was committed.
    Existing {
        /// The bench item.
        item: String,
        /// The point, by its index at the version the run landed on.
        point: u32,
        /// Whether a bench item already came from the point before the find;
        /// `item` is then that item, under the label it had.
        already: bool,
    },
    /// Built, put on the bench as `item`, and committed as the new point
    /// `point`.
    Committed {
        /// The bench item, seated on the point.
        item: String,
        /// The point the commit wrote.
        point: u32,
    },
    /// Built and put on the bench as `item`, and not committed: the run was
    /// asked not to commit (`why` is `None`), or the commit refused (`why`
    /// says why).
    OnBench {
        /// The bench item.
        item: String,
        /// The commit's refusal, when it refused.
        why: Option<String>,
    },
    /// Its track could not be built, so nothing went on the bench for it.
    NotBuilt {
        /// The build's refusal.
        why: String,
    },
}

impl Landing {
    /// The bench item it is, when it went on the bench.
    pub(crate) fn item(&self) -> Option<&str> {
        match self {
            Landing::Existing { item, .. }
            | Landing::Committed { item, .. }
            | Landing::OnBench { item, .. } => Some(item),
            Landing::NotBuilt { .. } => None,
        }
    }

    /// The point it is: the existing one, or the one the commit wrote.
    pub(crate) fn point(&self) -> Option<u32> {
        match self {
            Landing::Existing { point, .. } | Landing::Committed { point, .. } => Some(*point),
            Landing::OnBench { .. } | Landing::NotBuilt { .. } => None,
        }
    }
}

/// What one *Find Nearby Tracks* left behind, which the wire answers with.
#[derive(Debug, Clone)]
pub(crate) struct FoundNearby {
    /// What core found, every track and layer.
    pub(crate) found: NearbyTracks,
    /// Per usable track, in [`NearbyTracks::bench_order`], its index into
    /// `found.tracks` and what became of it.
    pub(crate) landed: Vec<(usize, Landing)>,
    /// Whether a version was pushed.
    pub(crate) changed: bool,
}

/// What a landing hands back to the background machinery: the row's outcome,
/// what the wire answers with, the row's kind, and the point to select once
/// the row is written.
pub(crate) struct Landed {
    /// The row's sentence, or the refusal.
    pub(crate) outcome: Result<String, String>,
    /// What the wire answers with.
    pub(crate) found: FoundNearby,
    /// `Edit` when the version wrote points, `Bench` when it wrote the bench
    /// alone or nothing.
    pub(crate) kind: Kind,
    /// The focused item's point, selected after the row, as a commit selects
    /// the point it wrote.
    pub(crate) select: Option<PointRef>,
}

/// "1 nearby track" or "8 nearby tracks".
fn tracks_phrase(n: usize) -> String {
    match n {
        1 => "1 nearby track".to_string(),
        n => format!("{n} nearby tracks"),
    }
}

impl AppState {
    /// Why *Find Nearby Tracks* cannot run on `image`, or `None` when it can.
    ///
    /// With `commit`, the reasons are *Create Track Here*'s, word for word:
    /// the node busy, the image not posed, or a node whose observations are
    /// `.sift` features, which a commit cannot write. Without it the last does
    /// not apply, since only the bench is written.
    pub(crate) fn find_nearby_tracks_refusal(
        &self,
        image: ImageRef,
        commit: bool,
    ) -> Option<String> {
        if commit {
            self.create_track_here_refusal(image)
        } else {
            self.posed_image_refusal(image)
        }
    }

    /// *Find Nearby Tracks* at `pixel` of `image`, from the Image Detail menu
    /// entry: the job started, or one failed row in the refusal's words.
    pub(crate) fn find_nearby_tracks_here(&mut self, image: ImageRef, pixel: [f32; 2]) {
        let pixel = [f64::from(pixel[0]), f64::from(pixel[1])];
        // The refusal is already the row `start_find_nearby_tracks` writes.
        let _ = self.start_find_nearby_tracks(image, pixel, true, None);
    }

    /// Find the tracks near `pixel` of `image` on a worker, then put them on
    /// the node's bench, and commit the new ones when `commit`, as one version
    /// when the worker comes home. `label` replaces the group label.
    ///
    /// What returns here is whether the run could **begin**. A refusal in
    /// front of the worker is one failed `Bench` row and no task.
    pub(crate) fn start_find_nearby_tracks(
        &mut self,
        image: ImageRef,
        pixel: [f64; 2],
        commit: bool,
        label: Option<&str>,
    ) -> Result<(), String> {
        let outcome = self
            .find_nearby_tracks_job(image, pixel, commit, label)
            .and_then(|job| {
                self.start_background_task(Operation::FIND_NEARBY_TRACKS, image.recon, job)
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
    /// **Only a `current` index file is handed over**, as
    /// [`AppState::create_track_at_pixel_job`] hands them, so a source whose
    /// file is missing or stale is skipped and named in the report. The grey
    /// images the far-field sweep and the layers sample are built on the
    /// worker, per run, from the decoded photographs: the viewer keeps no
    /// decoded set between runs for them to sit beside.
    pub(crate) fn find_nearby_tracks_job(
        &mut self,
        image: ImageRef,
        pixel: [f64; 2],
        commit: bool,
        label: Option<&str>,
    ) -> Result<Job, String> {
        if let Some(why) = self.find_nearby_tracks_refusal(image, commit) {
            return Err(why);
        }
        let id = image.recon;
        let image_name = self.image_name(image);
        let camera = self
            .image_camera(image)
            .ok_or_else(|| format!("{image_name} has no camera."))?;
        let (clamped, _) = bench::clamp_to_photograph(&camera, pixel);
        if clamped.is_some() {
            return Err(format!(
                "Cannot find nearby tracks at ({:.1}, {:.1}): that is not on the {}x{} \
                 photograph {image_name}.",
                pixel[0], pixel[1], camera.width, camera.height
            ));
        }
        self.refresh_index_files(id);
        let forest = (self.sift_index_state(id) == IndexFileState::Current)
            .then(|| self.sift_index(id).map(|index| Arc::clone(&index.forest)))
            .flatten();
        let cluster_patches = (self.cluster_patches_state(id) == IndexFileState::Current)
            .then(|| self.cluster_patches(id).map(|file| file.path.clone()))
            .flatten();

        let node = self
            .node(id)
            .ok_or_else(|| crate::state::NOT_LOADED.to_string())?;
        let recon = node.recon();
        let image_names: Vec<String> = recon
            .image_table
            .images
            .iter()
            .map(|row| row.name.clone())
            .collect();
        let sift_files: Vec<PathBuf> = (0..image_names.len())
            .map(|i| recon.sift_path_for_image(i))
            .collect();
        let edited = node.edited().clone();
        let every: Vec<usize> = (0..image_names.len()).collect();
        let views = self.view_sources_for(id, &every)?;
        let plan = Plan {
            image: image.index() as u32,
            pixel,
            image_name,
            commit,
            options: NearbyTrackOptions {
                label: label.map(str::to_string),
                ..NearbyTrackOptions::default()
            },
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
                        "Cannot find nearby tracks in {}: {e}",
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

    /// Land a finished run on the node at `index`: every usable track on the
    /// bench, the new ones committed, as one version; or, when nothing usable
    /// was found, the row that says so and no version.
    ///
    /// **One version for the whole find.** The bench puts and the commits are
    /// computed one after another on values that are never pushed, and only
    /// the last pair is: the reconstruction with every new point, and the
    /// bench with every track, the maps of the commits chained into the
    /// step's. An undo restores both halves to before the find and a redo
    /// restores them after it.
    ///
    /// **An existing point** goes on the bench as *Edit on Bench* puts it,
    /// through core's `create_track`, seated on the point, and is never
    /// committed. A point a bench item already came from keeps that item,
    /// under its own label, rather than having a second copy put on.
    ///
    /// **A label another item holds** takes core's `" (n)"` suffix, as every
    /// label on the bench does, so a second find at the same pixel puts its
    /// tracks beside the first's.
    ///
    /// The `1a` track, the nearest the pixel on the best-ranked layer, is the
    /// focused item afterwards, and its point, when it has one, the selection.
    pub(crate) fn land_nearby_tracks(&mut self, index: usize, run: NearbyTracksRun) -> Landed {
        let NearbyTracksRun {
            pixel,
            image_name,
            commit,
            found,
        } = run;
        let at = format!("({:.1}, {:.1}) in {image_name}", pixel[0], pixel[1]);
        let node = &self.scene[index];
        let id = node.id;
        let serial = node.history.current_version().serial;
        let mut bench = (**node.history.current_bench()).clone();
        let mut value = node.edited().clone();
        let mut maps = Vec::new();
        let mut created: Vec<u32> = Vec::new();
        let mut landed = Vec::new();
        let order = found.bench_order();

        for &k in &order {
            let track = &found.tracks[k];
            let label = track
                .label
                .clone()
                .expect("every track in the bench order is labelled");
            let landing = match (track.point, &track.track) {
                (Some(point), _) => {
                    // Already on the bench: the item there stands for it.
                    let node = &self.scene[index];
                    let already = bench.entries().iter().find_map(|entry| match &entry.item {
                        BenchItem::Track(t) => (self.resolved_origin(node, t) == Some(point))
                            .then(|| entry.label.clone()),
                    });
                    match already {
                        Some(item) => Landing::Existing {
                            item,
                            point,
                            already: true,
                        },
                        None => {
                            let options = CreateTrackOptions {
                                version: serial.as_u64(),
                                label: Some(label),
                            };
                            match bench::create_track(&bench, &value, point, &options) {
                                Ok((next, report)) => {
                                    bench = next;
                                    Landing::Existing {
                                        item: report.label,
                                        point,
                                        already: false,
                                    }
                                }
                                Err(e) => Landing::NotBuilt {
                                    why: format!("cannot put point {point} on the bench: {e}"),
                                },
                            }
                        }
                    }
                }
                (None, Some(Ok(built))) => {
                    let (next, item) = bench.put(&label, BenchItem::Track(Arc::new(built.clone())));
                    bench = next;
                    if !commit {
                        Landing::OnBench { item, why: None }
                    } else {
                        match bench::commit(&value, built) {
                            Ok((next, report)) => {
                                // Seated as `commit_bench_track` seats a
                                // creation: at the version the find was made
                                // on, by the index the point took, which the
                                // step's creation maps carry forward unchanged.
                                let seated = built.with_origin(serial.as_u64(), report.point);
                                bench = bench
                                    .replace(&item, BenchItem::Track(Arc::new(seated)))
                                    .expect("the item was just put on the bench");
                                value = next;
                                if report.replaced.is_none() {
                                    created.push(report.point);
                                }
                                maps.push(report.map);
                                Landing::Committed {
                                    item,
                                    point: report.point,
                                }
                            }
                            Err(e) => Landing::OnBench {
                                item,
                                why: Some(format!("cannot commit it: {e}")),
                            },
                        }
                    }
                }
                (None, Some(Err(why))) => Landing::NotBuilt { why: why.clone() },
                (None, None) => Landing::NotBuilt {
                    why: "no track was built for it".to_string(),
                },
            };
            landed.push((k, landing));
        }

        let usable = order.len();
        let layers = found.layers.len();
        let on_bench = landed.iter().filter(|(_, l)| l.item().is_some()).count();
        let group = found.group_label.clone();
        // What the tracks read as: the counts, the layers and the first
        // layer's confidence.
        let mut text = format!(
            "Found {} in {} at {at}",
            tracks_phrase(usable),
            match layers {
                1 => "1 layer".to_string(),
                n => format!("{n} layers"),
            },
        );
        if let Some(first) = found.first_layer().and_then(|l| l.ranking.as_ref()) {
            text.push_str(&format!(
                ", the first layer at {:.0}% confidence",
                first.confidence * 100.0
            ));
        }
        let count = |f: fn(&Landing) -> bool| landed.iter().filter(|(_, l)| f(l)).count();
        let committed = count(|l| matches!(l, Landing::Committed { .. }));
        let existing = count(|l| matches!(l, Landing::Existing { already: false, .. }));
        let already = count(|l| matches!(l, Landing::Existing { already: true, .. }));
        let kept = count(|l| matches!(l, Landing::OnBench { why: None, .. }));
        let refused = count(|l| matches!(l, Landing::OnBench { why: Some(_), .. }));
        let failed = count(|l| matches!(l, Landing::NotBuilt { .. }));
        let mut parts = Vec::new();
        if committed > 0 {
            parts.push(match committed {
                1 => "1 committed as a new point".to_string(),
                n => format!("{n} committed as new points"),
            });
        }
        if existing > 0 {
            parts.push(match existing {
                1 => "1 existing point put on the bench".to_string(),
                n => format!("{n} existing points put on the bench"),
            });
        }
        if already > 0 {
            parts.push(match already {
                1 => "1 existing point on the bench already".to_string(),
                n => format!("{n} existing points on the bench already"),
            });
        }
        if kept > 0 {
            parts.push(format!("{kept} put on the bench uncommitted"));
        }
        if refused > 0 {
            parts.push(format!("{refused} on the bench whose commit refused"));
        }
        if failed > 0 {
            parts.push(format!("{failed} that could not be built"));
        }

        if on_bench == 0 {
            // Nothing to put on the bench: no version, and the row says so.
            let text = if usable == 0 {
                let skipped: Vec<String> = found
                    .report
                    .sources
                    .iter()
                    .filter_map(|s| s.skipped.map(|what| format!("{} (no {what})", s.source)))
                    .collect();
                let mut text = format!("Found no nearby tracks at {at}");
                if !skipped.is_empty() {
                    text.push_str(&format!("; skipped {}", skipped.join(", ")));
                }
                text
            } else {
                format!("{text}: {}", parts.join(", "))
            };
            return Landed {
                outcome: Ok(text),
                found: FoundNearby {
                    found,
                    landed,
                    changed: false,
                },
                kind: Kind::Bench,
                select: None,
            };
        }

        // The `1a` track, or the first that went on the bench, is focused.
        let focus = landed
            .iter()
            .find_map(|(_, l)| l.item().map(|i| (i.to_string(), l.point())));
        let mut select = None;
        if let Some((item, point)) = &focus {
            select = point.map(|p| PointRef::new(id, p as usize));
            text.push_str(&format!(": {}; editing {item}", parts.join(", ")));
        }
        // Everything found was on the bench already: a version would change
        // nothing, so none is pushed, and the `1a` item is focused all the
        // same.
        if maps.is_empty() && bench == **self.scene[index].history.current_bench() {
            if let Some((item, _)) = &focus {
                self.focus_put_item(id, item);
            }
            return Landed {
                outcome: Ok(format!("{text}; no effect, the bench holds them already")),
                found: FoundNearby {
                    found,
                    landed,
                    changed: false,
                },
                kind: Kind::Bench,
                select,
            };
        }
        let version_label = format!("Found {} at {group}", tracks_phrase(on_bench));

        let node = &mut self.scene[index];
        let pushed = if maps.is_empty() {
            node.history.push_bench(Arc::new(bench), version_label)
        } else {
            let created = created_points(&value, &created);
            let map = match maps.len() {
                1 => maps.pop().expect("one map"),
                _ => PointMap::Chain(maps),
            };
            node.history.push_pair(
                Some(value),
                Arc::new(bench),
                None,
                map,
                version_label,
                created,
            )
        };
        let parent = crate::state::edits::version_before(node, pushed);
        if let Some((item, _)) = &focus {
            self.focus_put_item(id, item);
        }
        Landed {
            outcome: Ok(version_step_text(&text, parent, pushed)),
            found: FoundNearby {
                found,
                landed,
                changed: true,
            },
            kind: if committed > 0 {
                Kind::Edit
            } else {
                Kind::Bench
            },
            select,
        }
    }
}

/// The points one find created, named by the content hash of the edit that
/// created them all, in the order it created them.
fn created_points(value: &EditedReconstruction, indexes: &[u32]) -> Option<CreatedPoints> {
    if indexes.is_empty() {
        return None;
    }
    let records: Vec<_> = indexes
        .iter()
        .map(|&i| value.point(i).map(|p| p.to_record()))
        .collect::<Option<_>>()?;
    value
        .point_edit_hash(&records)
        .ok()
        .map(|hash| CreatedPoints {
            hash,
            indexes: indexes.to_vec(),
        })
}

/// Everything the worker reads beside the views and the value, owned.
struct Plan {
    image: u32,
    pixel: [f64; 2],
    image_name: String,
    commit: bool,
    options: NearbyTrackOptions,
    /// The SIFT index, when it is current.
    forest: Option<Arc<LazyKdForestU8>>,
    /// Every image's `.sift` path, in the node's order.
    sift_files: Vec<PathBuf>,
    /// The cluster-patches file, when it is current.
    cluster_patches: Option<PathBuf>,
    /// Every image's name, in the node's order, which the clusters are
    /// indexed onto.
    image_names: Vec<String>,
}

/// Every image's `.sift` keypoints and descriptors, when every file is on
/// disk and reads; `None` otherwise, and guided matching and the
/// constellation source are then skipped.
fn read_sift(
    files: &[PathBuf],
    progress: &Progress<'_>,
) -> Option<(Vec<ImageKeypoints>, Vec<ImageDescriptors>)> {
    if files.is_empty() || !files.iter().all(|path| path.is_file()) {
        return None;
    }
    let _phase = progress.phase("read .sift files");
    let read: Result<Vec<_>, String> = files
        .par_iter()
        .map(|path| {
            sfmtool_sift_format::read_sift_features(path)
                .map_err(|e| format!("{}: {e}", path.display()))
        })
        .collect();
    match read {
        Ok(read) => Some(
            read.into_iter()
                .map(|f| {
                    (
                        ImageKeypoints {
                            positions: f.positions_xy,
                            affine_shapes: f.affine_shapes,
                        },
                        ImageDescriptors::new(f.descriptors),
                    )
                })
                .unzip(),
        ),
        Err(e) => {
            sfmtool_core::progress_warn!(
                progress,
                "Cannot read the .sift files ({e}); guided matching and the constellation \
                 source are skipped"
            );
            None
        }
    }
}

/// The worker's half after the decode: read what the sources need, find the
/// tracks, and hand them back.
fn run(
    plan: Plan,
    edited: &EditedReconstruction,
    views: &[sfmtool_core::patch::normal_refine::ProjectedImage<'_>],
    progress: &Progress<'_>,
) -> Finished {
    let sift = read_sift(&plan.sift_files, progress);
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
    let rays = KeypointRays::new(views.len());
    let grey = GreyImages::new(views.len());
    let sources = NearbyTrackSources {
        clusters: clusters.as_ref(),
        guided: sift.as_ref().map(|(keypoints, descriptors)| GuidedSource {
            keypoints,
            descriptors,
            rays: &rays,
        }),
        sift_index: match (&plan.forest, &sift) {
            (Some(forest), Some((keypoints, _))) => Some(SiftIndexSource { forest, keypoints }),
            _ => None,
        },
    };
    let found = find_nearby_tracks(
        edited,
        views,
        &grey,
        &sources,
        plan.image,
        plan.pixel,
        &plan.options,
        progress,
    );
    match found {
        Ok(found) => Finished::NearbyTracks(Box::new(NearbyTracksRun {
            pixel: plan.pixel,
            image_name: plan.image_name,
            commit: plan.commit,
            found,
        })),
        Err(NearbyTracksError::Cancelled) => Finished::Cancelled,
        Err(e) => Finished::Failed(format!(
            "Cannot find nearby tracks at ({:.1}, {:.1}) in {}: {e}",
            plan.pixel[0], plan.pixel[1], plan.image_name
        )),
    }
}
