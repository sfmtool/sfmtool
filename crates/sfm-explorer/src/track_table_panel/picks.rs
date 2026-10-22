// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The Track Table's **reference picks**: the reference observation of each
//! visible point that stores none, picked by the reference-view rule off the
//! GUI thread, so the *Crop* column knows which photograph to cut.
//!
//! See `specs/gui/track-table.md` § "The reference photograph". The picks are
//! a **quiet task** in the fast queue
//! ([`Operation::TRACK_TABLE_REFERENCE_PICKS`],
//! `specs/gui/background-tasks.md` § "Quiet tasks"), and their driver follows
//! the table job's ([`super::job`]):
//!
//! - **Its inputs** are the points the panel draws that need a pick, for the
//!   node it shows and the document serial at that node's cursor. The panel
//!   names them each frame its scroll position has stopped changing
//!   ([`AppState::want_reference_picks`]), and the frame clears them before
//!   the dock draws ([`AppState::hide_track_table`]), so a hidden panel or one
//!   being scrolled asks for nothing.
//! - **One task works through them one pick at a time.** It takes the next
//!   point from a list the GUI thread shares with it ([`PickQueue`]), decodes
//!   the photographs of the point's track through the photograph cache and
//!   runs the rule ([`pick_reference`]), sends the pick back, and takes the
//!   next. Each frame the driver puts the points the panel wants and the
//!   picks do not hold in that list, so a row that leaves the view is taken
//!   off it, and the task ends when the list is empty.
//! - **An edit cancels it.** When the document serial at its node's cursor
//!   moves, or the node is closed, the task is asked to stop through its
//!   `Progress`, and a pick that comes back for a serial the node has left is
//!   dropped. Nothing is queued for the new version until the panel's rows
//!   are of it.
//! - **It does not wait for its node.** A pick is kept by the node, the point
//!   and the serial it was made for ([`ReferencePicks`]), not written into the
//!   version, so the task runs on a node a slow task holds, against the version
//!   on screen ([`crate::background::QuietSource::waits_for_its_node`]). It
//!   gives way to any task queued in the fast queue, and the driver queues it
//!   again behind that task with the points still to pick.

use std::collections::{HashMap, VecDeque};
use std::sync::mpsc::{Receiver, Sender, TryRecvError};
use std::sync::{Arc, Mutex};

use sfmtool_core::camera::PhotographCache;
use sfmtool_core::geometry::RigidTransform;
use sfmtool_core::patch::keypoint_subpixel::KeypointSubpixelParams;
use sfmtool_core::patch::normal_refine::ProjectedImage;
use sfmtool_core::patch::stored_bitmap::render_reference;
use sfmtool_core::progress::Progress;
use sfmtool_core::EditedReconstruction;

use crate::background::{Operation, QuietJob, QuietRank, QuietSource};
use crate::document::VersionSerial;
use crate::scene::ReconId;
use crate::state::edits::DECODE_AND_STEP;
use crate::state::AppState;

/// One pick, named by the version it was made for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct PickKey {
    node: ReconId,
    point: u32,
    document: VersionSerial,
}

/// The points the panel wants picks for in this frame.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Wanted {
    node: ReconId,
    document: VersionSerial,
    points: Vec<u32>,
}

/// The list of points the running task works through, shared with the GUI
/// thread, which replaces it each frame with the points still wanted.
#[derive(Debug, Default)]
struct PickQueue {
    /// The points still to pick, the first first.
    points: VecDeque<u32>,
    /// The point the task is picking now, which the GUI thread leaves out of
    /// `points` so it is not picked twice.
    in_hand: Option<u32>,
}

/// What comes back on the task's channel.
enum Message {
    /// The reference picked for `point`: an index into its track, or `None`
    /// where the rule picked none.
    Picked { point: u32, reference: Option<u32> },
    /// The queue dropped the task before it started.
    Dropped,
}

/// The task the driver has queued, waiting or running.
struct Queued {
    node: ReconId,
    document: VersionSerial,
    /// The quiet task it runs as.
    task: u64,
    /// The points it works through.
    queue: Arc<Mutex<PickQueue>>,
    /// Where it sends its picks. The channel ends when the task's work has
    /// returned or been dropped.
    messages: Receiver<Message>,
}

/// The reference picks' state on [`AppState`]: the picks made, held by the
/// Track Table and not by any version, what the panel wants this frame, and
/// the task queued for it.
#[derive(Default)]
pub(crate) struct ReferencePicks {
    /// Every pick made for a version still at its node's cursor.
    held: HashMap<PickKey, Option<u32>>,
    /// The document serial the picks held for each node were made for, so a
    /// frame finds out whether any are out of date by looking at the nodes
    /// rather than at every pick.
    serials: HashMap<ReconId, VersionSerial>,
    /// The points the panel drew this frame that need a pick, or `None`
    /// while it is hidden, scrolling or drawing rows of a version the node
    /// has left.
    wanted: Option<Wanted>,
    /// The task queued, waiting or running, if there is one.
    queued: Option<Queued>,
    /// A gate the next task's work waits at before it starts, so a test can
    /// act while it is known to be running.
    #[cfg(test)]
    pub(super) hold_next: Option<Receiver<()>>,
    /// How many tasks have been queued, for the tests.
    #[cfg(test)]
    pub(super) tasks_queued: usize,
}

impl ReferencePicks {
    /// The id of the quiet task queued, if there is one.
    #[cfg(test)]
    pub(super) fn queued_task(&self) -> Option<u64> {
        self.queued.as_ref().map(|queued| queued.task)
    }

    /// The points the queued task still has to pick, and the one it is
    /// picking.
    #[cfg(test)]
    pub(super) fn queued_points(&self) -> Option<(Vec<u32>, Option<u32>)> {
        let queued = self.queued.as_ref()?;
        let queue = queued.queue.lock().unwrap_or_else(|e| e.into_inner());
        Some((queue.points.iter().copied().collect(), queue.in_hand))
    }

    /// The points the panel asked for picks for this frame.
    #[cfg(test)]
    pub(super) fn wanted_points(&self) -> Option<&[u32]> {
        self.wanted.as_ref().map(|wanted| wanted.points.as_slice())
    }
}

/// The reference the reference-view rule picks for `point` of `edited`, at its
/// stored keypoints: an index into its track, `Some(None)` where the rule
/// picks none, and `None` when `progress` was cancelled.
///
/// The photographs of the point's track are read through `photographs`, in the
/// first share of `progress`'s range ([`DECODE_AND_STEP`]); a photograph that
/// cannot be read, or is not the size its camera says, is left out of the
/// views, as the open's display render leaves it out. The rule runs in the
/// second share, over the views left, through
/// `stored_bitmap::render_reference` at the reconstruction's patch resolution
/// ([`crate::bench::patch_resolution`]): the pick an evaluation would make
/// without localising the rows first. With fewer than two views there is no
/// pick, as for the bitmap writers. The patch frame keeps the point's `w`, so a
/// point at infinity is picked for as any other.
pub(super) fn pick_reference(
    edited: &EditedReconstruction,
    photographs: &PhotographCache,
    point: u32,
    progress: &Progress<'_>,
) -> Option<Option<u32>> {
    let Some(view) = edited.point(point) else {
        return Some(None);
    };
    let Some(patch) = view.patch_frame() else {
        return Some(None);
    };
    let recon = &edited.base;
    let [decode, rule] = progress.split(DECODE_AND_STEP);
    let observations = view.observations();
    // The observation each view is, its keypoint, its pose and its pyramid.
    let mut found = Vec::with_capacity(observations.len());
    for (k, observation) in observations.iter().enumerate() {
        if decode.is_cancelled() {
            return None;
        }
        let index = observation.image_index as usize;
        let read = (|| {
            let keypoint = view.keypoint_xy(k)?;
            let image = recon.image_set.images.get(index)?;
            let camera = recon.image_set.cameras.get(image.camera_index as usize)?;
            let pyramid = photographs.get(&crate::state::photograph_path(recon, index)?)?;
            let level = pyramid.level(0);
            (level.width() == camera.width && level.height() == camera.height).then_some((
                k,
                camera,
                crate::scene::cam_from_world(image),
                [f64::from(keypoint[0]), f64::from(keypoint[1])],
                pyramid,
            ))
        })();
        found.extend(read);
        decode.count(k as u64 + 1, Some(observations.len() as u64), "photographs");
    }
    if found.len() < 2 {
        return Some(None);
    }
    let poses: Vec<RigidTransform> = found.iter().map(|f| f.2.clone()).collect();
    let views: Vec<ProjectedImage<'_>> = found
        .iter()
        .zip(&poses)
        .map(|(f, cam_from_world)| ProjectedImage {
            camera: f.1,
            cam_from_world,
            pyramid: &f.4,
        })
        .collect();
    let patch_views: Vec<u32> = (0..views.len() as u32).collect();
    let keypoints: Vec<Option<[f64; 2]>> = found.iter().map(|f| Some(f.3)).collect();
    let params = KeypointSubpixelParams {
        resolution: crate::bench::patch_resolution(recon),
        ..KeypointSubpixelParams::default()
    };
    let render = render_reference(
        &patch,
        &views,
        &patch_views,
        &keypoints,
        params.resolution,
        params.sampler,
        &params.reference_view,
        &rule,
    );
    if rule.is_cancelled() {
        return None;
    }
    Some(render.reading.choice.reference.map(|r| found[r].0 as u32))
}

/// The work one task does: take the next point off `queue`, pick its
/// reference, send it, and go on until the list is empty or the task is
/// asked to stop. Each pick spans the task's whole range, since the list
/// changes as the rows scroll; nothing draws a quiet task's fraction.
fn work(
    edited: EditedReconstruction,
    photographs: Arc<PhotographCache>,
    queue: Arc<Mutex<PickQueue>>,
    sender: Sender<Message>,
    wake: Option<Arc<dyn Fn() + Send + Sync>>,
) -> impl for<'a> FnOnce(&Progress<'a>) + Send + 'static {
    move |progress: &Progress<'_>| loop {
        if progress.is_cancelled() {
            return;
        }
        let point = {
            let mut queue = queue.lock().unwrap_or_else(|e| e.into_inner());
            let Some(point) = queue.points.pop_front() else {
                queue.in_hand = None;
                return;
            };
            queue.in_hand = Some(point);
            point
        };
        let Some(reference) = pick_reference(&edited, &photographs, point, progress) else {
            return;
        };
        let _ = sender.send(Message::Picked { point, reference });
        if let Some(wake) = wake.as_ref() {
            wake();
        }
    }
}

/// Whether a task's work, run with the cancel flag already set over a list
/// of points, stops without sending a pick: the test of
/// [`Operation::TRACK_TABLE_REFERENCE_PICKS`]'s claim to be cancellable.
#[cfg(test)]
pub(crate) fn job_stops_when_cancelled() -> bool {
    let edited = sfmtool_core::EditedReconstruction::new(std::sync::Arc::new(
        crate::state::edits::tests::projected_embedded_demo(12),
    ));
    let queue = Arc::new(Mutex::new(PickQueue {
        points: (0..12).collect(),
        in_hand: None,
    }));
    let (sender, messages) = std::sync::mpsc::channel();
    let cancel = std::sync::atomic::AtomicBool::new(true);
    let photographs = Arc::new(PhotographCache::new(0, 1));
    work(edited, photographs, Arc::clone(&queue), sender, None)(
        &Progress::none().cancelled_by(&cancel),
    );
    let untouched = queue.lock().unwrap_or_else(|e| e.into_inner()).points.len() == 12;
    messages.try_recv().is_err() && untouched
}

impl AppState {
    /// Say that the Track Table panel draws `points` of `node`, at the
    /// version whose document serial is `document`, and that each needs a
    /// reference pick. Called on a frame whose scroll position has stopped
    /// changing, for rows of the version at the node's cursor.
    pub(super) fn want_reference_picks(
        &mut self,
        node: ReconId,
        document: VersionSerial,
        points: Vec<u32>,
    ) {
        self.reference_picks.wanted = Some(Wanted {
            node,
            document,
            points,
        });
    }

    /// Forget what the panel wanted picked: called by the frame before the
    /// dock draws, with [`AppState::hide_track_table`].
    pub(super) fn hide_reference_picks(&mut self) {
        self.reference_picks.wanted = None;
    }

    /// The reference picked for `point` of `node` at `document`: an index
    /// into its track, `Some(None)` where the rule picked none, or `None`
    /// where no pick is held.
    pub(super) fn reference_pick(
        &self,
        node: ReconId,
        point: u32,
        document: VersionSerial,
    ) -> Option<Option<u32>> {
        self.reference_picks
            .held
            .get(&PickKey {
                node,
                point,
                document,
            })
            .copied()
    }

    /// The document serial at `node`'s cursor, or `None` when it is closed.
    fn document_at_cursor(&self, node: ReconId) -> Option<VersionSerial> {
        self.node(node)
            .map(|node| node.history.current_version().document_serial)
    }

    /// Take the picks the queued task has made, forget the picks of versions
    /// their nodes have left, and queue, update or stop the task so that it
    /// works through the points the panel wants that have no pick.
    ///
    /// Called twice per frame, after [`AppState::drive_track_table_columns`].
    /// Returns whether anything changed that the next frame has to draw.
    pub(crate) fn drive_track_table_picks(&mut self) -> bool {
        let mut changed = self.poll_reference_picks();
        // A pick is for one version; the rows of another never read it. The
        // picks are pruned only on a frame a node's serial has moved, so the
        // cost of a frame does not grow with the picks held.
        let left: Vec<ReconId> = self
            .reference_picks
            .serials
            .iter()
            .filter(|&(&node, &document)| self.document_at_cursor(node) != Some(document))
            .map(|(&node, _)| node)
            .collect();
        if !left.is_empty() {
            let picks = &mut self.reference_picks;
            for node in &left {
                picks.serials.remove(node);
            }
            picks.held.retain(|key, _| !left.contains(&key.node));
        }

        // The points still to pick, for the version at the node's cursor.
        let todo = self.reference_picks.wanted.as_ref().and_then(|wanted| {
            if self.document_at_cursor(wanted.node) != Some(wanted.document) {
                return None;
            }
            let points: Vec<u32> = wanted
                .points
                .iter()
                .copied()
                .filter(|&point| {
                    !self.reference_picks.held.contains_key(&PickKey {
                        node: wanted.node,
                        point,
                        document: wanted.document,
                    })
                })
                .collect();
            (!points.is_empty()).then_some((wanted.node, wanted.document, points))
        });

        if let Some(queued) = self.reference_picks.queued.as_ref() {
            if self.document_at_cursor(queued.node) != Some(queued.document) {
                // An edit, an undo or a close: its picks would be for a
                // version the rows have left. It is forgotten once it has
                // reported back.
                let task = queued.task;
                self.cancel_quiet_task(task);
                return changed;
            }
            let same = todo.as_ref().is_some_and(|(node, document, _)| {
                (*node, *document) == (queued.node, queued.document)
            });
            let running = self.quiet_task_running(queued.task);
            if same || running {
                // The points still wanted replace its list; a row that left
                // the view leaves it, and with none wanted it ends after the
                // pick in hand.
                let mut queue = queued.queue.lock().unwrap_or_else(|e| e.into_inner());
                let in_hand = queue.in_hand;
                queue.points = match (&todo, same) {
                    (Some((_, _, points)), true) => points
                        .iter()
                        .copied()
                        .filter(|&p| Some(p) != in_hand)
                        .collect(),
                    _ => VecDeque::new(),
                };
                return changed;
            }
            // Waiting and no longer wanted.
            let task = queued.task;
            self.reference_picks.queued = None;
            self.cancel_quiet_task(task);
            changed = true;
        }
        if let Some((node, document, points)) = todo {
            if self.queue_reference_picks(node, document, points) {
                changed = true;
            }
        }
        changed
    }

    /// Queue a *Track Table reference picks* quiet task in the fast queue
    /// working through `points` of `node` at `document`. False when its
    /// worker could not be spawned; the next frame tries again.
    fn queue_reference_picks(
        &mut self,
        node: ReconId,
        document: VersionSerial,
        points: Vec<u32>,
    ) -> bool {
        let Some(scene_node) = self.node(node) else {
            return false;
        };
        let edited = scene_node.edited().clone();
        let label = scene_node.label.clone();
        let queue = Arc::new(Mutex::new(PickQueue {
            points: points.into(),
            in_hand: None,
        }));
        let (sender, messages) = std::sync::mpsc::channel();
        let dropped = sender.clone();
        let work = work(
            edited,
            Arc::clone(&self.photographs),
            Arc::clone(&queue),
            sender,
            self.wake.clone(),
        );
        #[cfg(test)]
        let hold = self.reference_picks.hold_next.take();
        let run = move |progress: &Progress<'_>| {
            #[cfg(test)]
            if let Some(hold) = hold {
                let _ = hold.recv();
            }
            work(progress);
        };
        let started = self.start_quiet_task(
            Operation::TRACK_TABLE_REFERENCE_PICKS,
            QuietSource::TrackTableReferencePicks,
            QuietRank::TrackTableReferencePicks,
            node,
            label,
            QuietJob::new(run, move || {
                let _ = dropped.send(Message::Dropped);
            }),
        );
        let Ok(task) = started else {
            return false;
        };
        #[cfg(test)]
        {
            self.reference_picks.tasks_queued += 1;
        }
        self.reference_picks.queued = Some(Queued {
            node,
            document,
            task,
            queue,
            messages,
        });
        true
    }

    /// Take every pick the queued task has sent, keeping each that is for its
    /// node's current serial, and forget the task once it has ended or was
    /// dropped.
    fn poll_reference_picks(&mut self) -> bool {
        let mut changed = false;
        loop {
            let Some(queued) = self.reference_picks.queued.as_ref() else {
                return changed;
            };
            let (node, document) = (queued.node, queued.document);
            match queued.messages.try_recv() {
                Ok(Message::Picked { point, reference }) => {
                    if self.document_at_cursor(node) == Some(document) {
                        self.reference_picks.serials.insert(node, document);
                        self.reference_picks.held.insert(
                            PickKey {
                                node,
                                point,
                                document,
                            },
                            reference,
                        );
                        changed = true;
                    }
                }
                Ok(Message::Dropped) | Err(TryRecvError::Disconnected) => {
                    self.reference_picks.queued = None;
                    return true;
                }
                Err(TryRecvError::Empty) => return changed,
            }
        }
    }

    /// Drive the picks until the panel's wanted points are picked and the
    /// queued task has ended, stepping the fast queue meanwhile: the test
    /// seam, as `settle_track_table_columns` is for the table job.
    #[cfg(test)]
    pub(super) fn settle_reference_picks(&mut self) {
        loop {
            self.drive_track_table_picks();
            if self.reference_picks.queued.is_none() {
                self.land_quiet_tasks();
                return;
            }
            self.wait_for_reference_picks();
        }
    }

    /// Block until the queued task has ended or was dropped, stepping the
    /// fast queue meanwhile and keeping the picks it sends, without driving:
    /// its list stays as it was.
    #[cfg(test)]
    pub(super) fn wait_for_reference_picks(&mut self) {
        loop {
            let Some(task) = self.reference_picks.queued.as_ref().map(|q| q.task) else {
                return;
            };
            self.poll_reference_picks();
            if self.reference_picks.queued.is_none() {
                // Its channel ended: let the queue land the task.
                while self.quiet_task_running(task) {
                    self.step_queue(crate::background::Queue::Fast);
                }
                return;
            }
            if !self.step_queue(crate::background::Queue::Fast) {
                std::thread::yield_now();
            }
        }
    }
}
