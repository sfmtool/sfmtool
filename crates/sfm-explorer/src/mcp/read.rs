// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The reads that answer a question about one entity or about the session, and
//! change nothing.
//!
//! (`get_scene` is [`super::render::scene`] directly — it has no arguments to
//! resolve and nothing to look up — and the layout read is
//! [`super::layout::get_window_layout`], which needs a window host.)
//!
//! Each takes `&mut AppState` rather than `&AppState`, which for a read looks
//! wrong and is not: resolving a `.sfmr` observation to a pixel means reading
//! the `.sift` file it points into, and the viewer memoizes that in
//! `AppState::sift_cache`. Reading through the cache is what makes the number
//! reported here the same number the Point Track panel shows, rather than a
//! second implementation of it.

use std::time::Duration;

use serde_json::{json, Value};
use sfmtool_core::progress::{Level, Progress};

use super::{
    render, resolve_camera_image, resolve_camera_intrinsics, resolve_point, CameraImageSel,
    JsonReply, ToolError,
};
use crate::action_log::{ActionLog, Actor, Breakdown};
use crate::progress::{Count, Detail};
use crate::scene::{point_id, ImageRef};
use crate::state::{ensure_sift_cached, AppState};

/// How many camera images one `list_camera_images` call will return, whatever
/// it asked for.
///
/// The surface is not a data channel: a caller that wants every row of a
/// thousand-image reconstruction should page, and one that wants the data
/// should read the `.sfmr`.
pub(super) const MAX_LIMIT: usize = 500;

/// The default page, when the call names no limit.
pub(super) const DEFAULT_LIMIT: usize = 50;

/// How many Action Log entries one `get_action_log` returns by default.
///
/// Larger than a camera-image page because the log is read as a transcript:
/// an agent that stepped away wants the run of what happened, not a window
/// onto it, and an entry is a line rather than a record.
pub(super) const ACTION_LOG_DEFAULT_LIMIT: usize = 200;

/// The most one `get_action_log` will return, whatever it asked for.
pub(super) const ACTION_LOG_MAX_LIMIT: usize = 1_000;

/// `get_action_log`: what happened from a revision onward, oldest first.
///
/// The log's own clock rather than a timestamp decides what "onward" means
/// (`ActionLog::since`), because two entries can share an instant and a
/// coalescing fold moves an entry's time — so a reader sent a revision back is
/// told about the fold as well as about the new lines.
pub(super) fn get_action_log(
    state: &mut AppState,
    since_revision: u64,
    limit: usize,
    actors: &[Actor],
    detail: bool,
) -> JsonReply {
    let log = &state.action_log;
    let limit = limit.min(ACTION_LOG_MAX_LIMIT);
    let mut matching = log
        .since(since_revision)
        .filter(|entry| actors.contains(&entry.actor));
    let entries: Vec<Value> = matching
        .by_ref()
        .take(limit)
        .map(|e| entry(log, e, detail))
        .collect();
    Ok(json!({
        "revision": log.revision(),
        "oldest_revision": log.oldest_revision(),
        // Asked of the iterator rather than counted: whether anything is left
        // is the whole question, and counting the rest would walk entries the
        // reply does not carry.
        "truncated": matching.next().is_some(),
        "entries": entries,
    }))
}

/// One Action Log row on the wire.
///
/// `tool` rides beside `kind` on a query row rather than inside it, so that
/// `kind` stays a closed vocabulary while the tool table grows.
///
/// `detail` is what the call asked for: the breakdown is several times the
/// size of the row it hangs off, and an agent reading the log to find out what
/// happened does not want it.
fn entry(log: &ActionLog, entry: &crate::action_log::Entry, detail: bool) -> Value {
    let mut row = json!({
        "revision": entry.revision,
        "at": log.format_rfc3339(entry.at),
        "actor": entry.actor.wire_name(),
        "kind": entry.kind.wire_name(),
        "failed": entry.failed,
        "text": entry.text,
    });
    let fields = row.as_object_mut().expect("a log row is an object");
    // Absent rather than null while an action has not been drawn yet, and
    // absent forever on a row whose run folded under it. A reader that wants
    // the number can ask again; one that does not is not handed a null to
    // special-case.
    if let Some(took) = entry.took {
        fields.insert("took_ms".into(), json!(milliseconds(took)));
    }
    if detail {
        // Beside `took_ms` and absent for the same reason it is: an entry with
        // no cost yet has no account to close. It is what makes the breakdown
        // reconcile with the headline, so the stage nobody has named reads as
        // a gap rather than as silence.
        if let Some(elsewhere) = ActionLog::elsewhere(entry) {
            fields.insert("elsewhere_ms".into(), json!(milliseconds(elsewhere)));
        }
        fields.insert("detail".into(), breakdown(&Breakdown::of(entry)));
    }
    if let crate::action_log::Kind::Query(tool) = entry.kind {
        fields.insert("tool".into(), json!(tool));
    }
    row
}

/// One breakdown as an array of rows, in the order the panel draws them.
///
/// The one spelling both readers of a breakdown use: an Action Log entry's,
/// and a background operation's while it is still reporting. Two spellings
/// would be two things that can drift, and a reader comparing a solve it
/// watched against the row it left would be comparing two claims.
fn breakdown(breakdown: &Breakdown<'_>) -> Value {
    let rows: Vec<Value> = ActionLog::detail_in_draw_order(breakdown)
        .map(detail_row)
        .collect();
    json!(rows)
}

/// One row of an entry's breakdown: a stage and what it cost, or something the
/// operation said.
///
/// `cpu_ms`, `note` and `runs` are carried only where they apply, which is the
/// same thing the panel does with the columns it draws them in: a stage that
/// ran once carries no count, and one that reported no thread-summed time
/// carries no CPU figure rather than a zero.
fn detail_row(detail: &Detail) -> Value {
    match detail {
        Detail::Phase {
            name,
            depth,
            took,
            cpu,
            note,
            note_last,
            runs,
        } => {
            let mut row = json!({
                "kind": "phase",
                "name": name,
                "depth": depth,
                "ms": milliseconds(*took),
            });
            let fields = row.as_object_mut().expect("a detail row is an object");
            if let Some(cpu) = cpu {
                fields.insert("cpu_ms".into(), json!(milliseconds(*cpu)));
            }
            // Both ends where a folded row's runs disagreed, through the
            // panel's own spelling of it: one end on its own is a claim about
            // the row that only one of its runs supports.
            if let Some(note) = ActionLog::note_text(note.as_deref(), note_last.as_deref()) {
                fields.insert("note".into(), json!(note));
            }
            if *runs > 1 {
                fields.insert("runs".into(), json!(runs));
            }
            row
        }
        Detail::Message { level, depth, text } => json!({
            "kind": "message",
            "level": match level {
                Level::Info => "info",
                Level::Warn => "warn",
            },
            "depth": depth,
            "text": text,
        }),
    }
}

/// A duration as the wire spells one: milliseconds, fractional.
fn milliseconds(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

/// `get_background_process`: what the viewer is busy with, or what it was busy
/// with last.
///
/// One shape for both, discriminated by `running`, because the question an
/// agent brings here is a single one asked at an unknown moment: "the solve I
/// started -- is it still going, and what has it cost?" A tool that answered
/// only about a live operation would have to be paired with a second one for
/// the answer, and the agent would have to know which to call before knowing
/// whether it had finished.
pub(super) fn get_background_process(state: &AppState) -> JsonReply {
    let Some(process) = state.background() else {
        return Ok(finished_operation(state));
    };
    let live = process.collector.live();
    let mut reply = background_summary(state).expect("something is running");
    let fields = reply.as_object_mut().expect("the block is an object");
    fields.insert("cancellable".into(), json!(process.operation.cancellable));
    // Absent rather than null, as every optional field on this surface is: a
    // kernel that reports no count has not reported a count of nothing.
    if let Some(count) = process.collector.count() {
        fields.insert("progress".into(), progress(count));
    }
    if let Some(status) = process.collector.status() {
        fields.insert("status".into(), json!(status));
    }
    // The innermost stage that has been entered and not left, which is what
    // the panel's spinner names. A message row is not a phase, so a `live.open`
    // index landing on one leaves this out rather than reporting its text as a
    // stage.
    if let Some(Detail::Phase { name, .. }) = live.open.last().and_then(|&row| live.rows.get(row)) {
        fields.insert("phase".into(), json!(name));
    }
    fields.insert(
        "phases".into(),
        breakdown(&Breakdown::running(&ActionLog::capped_recent(live.rows))),
    );
    Ok(reply)
}

/// The same reply, for a session with nothing running.
///
/// The last operation rather than nothing at all, for the reason the Background
/// panel's idle form shows it: the agent that took a handle comes back for the
/// answer after the operation is gone, and "what did it cost" is the question
/// it brings.
fn finished_operation(state: &AppState) -> Value {
    let Some(last) = state.last_background.as_ref() else {
        // Both discriminators present, so a reader never has to tell a missing
        // key from a false one: this session has run nothing at all.
        return json!({ "running": false, "finished": false });
    };
    let (failed, text) = match &last.outcome {
        Ok(text) => (false, text),
        Err(message) => (true, message),
    };
    json!({
        "running": false,
        "finished": true,
        "operation": last.operation.name,
        "reconstruction_label": last.label,
        "operation_id": last.id,
        // The same field as a running operation's, which is what makes one
        // shape out of two: seconds so far while it runs, seconds in total
        // once it is over.
        "elapsed_s": last.took.as_secs_f64(),
        // The Action Log's own two fields for how a row ended, in its words,
        // so the sentence here and the sentence there are one sentence.
        "failed": failed,
        "text": text,
        "phases": breakdown(&Breakdown::running(&ActionLog::capped_recent(last.detail.clone()))),
    })
}

/// The `background` block `get_scene` carries, or `None` with nothing running.
///
/// **Deliberately not the whole of `get_background_process`.** `get_scene` is
/// the most-polled tool on this surface, and the reason this block is here at
/// all is that an agent polling it should learn the viewer is busy without a
/// second call. That question is answered by a handful of scalars whose size
/// does not depend on the operation. The phase table does depend on it -- a
/// three-round, sixty-iteration adjustment reports hundreds of rows -- so it
/// belongs to the tool an agent asks when it wants it, and the open phase and
/// the status line go with it because they are narrative rather than something
/// a caller acts on.
///
/// `None` rather than the last operation, for the same reason: a block that
/// went on describing a solve that ended half an hour ago would be carried by
/// every poll for the rest of the session, and `background != null` would stop
/// meaning "the viewer is busy", which is the one thing this block is read for.
pub(super) fn background_summary(state: &AppState) -> Option<Value> {
    let process = state.background()?;
    let mut block = json!({
        // The discriminator `bundle_adjust`'s handle uses, so the two replies
        // an agent sees about one operation are read the same way.
        "running": true,
        "operation": process.operation.name,
        "reconstruction_label": process.label,
        "operation_id": process.id,
        "elapsed_s": process.started.elapsed().as_secs_f64(),
    });
    // One number for "how far along", rather than the kernel's own count,
    // which needs three fields and a unit only that kernel defines. It is the
    // mapped sum of what the stages reported and is therefore measured: a
    // silent stage does not move it, and nothing here interpolates across one.
    if let Some(fraction) = process.collector.fraction() {
        block
            .as_object_mut()
            .expect("the block is an object")
            .insert("fraction".into(), json!(fraction));
    }
    Some(block)
}

/// A kernel's own count of what it is through: `{ done, total, unit }`, with
/// `total` absent where the operation does not know how many there are.
fn progress(count: Count) -> Value {
    let mut block = json!({ "done": count.done, "unit": count.unit });
    if let Some(total) = count.total {
        block
            .as_object_mut()
            .expect("the block is an object")
            .insert("total".into(), json!(total));
    }
    block
}

pub(super) fn list_camera_images(
    state: &mut AppState,
    reconstruction_label: Option<&str>,
    offset: usize,
    limit: usize,
) -> JsonReply {
    let id = super::resolve_reconstruction(state, reconstruction_label)?;
    let node = state.node(id).expect("just resolved");
    let recon = node.recon();
    let total = recon.image_table.images.len();
    let limit = limit.min(MAX_LIMIT);
    let end = offset.saturating_add(limit).min(total);
    let observations = render::observations_per_image(recon);
    let rows: Vec<Value> = (offset.min(total)..end)
        .map(|index| render::camera_image_row(recon, index, observations[index]))
        .collect();
    Ok(json!({
        "reconstruction_label": node.label,
        "total": total,
        "offset": offset,
        "camera_images": rows,
    }))
}

pub(super) fn get_camera_image(
    state: &mut AppState,
    reconstruction_label: Option<&str>,
    selector: &CameraImageSel,
) -> JsonReply {
    let id = super::resolve_reconstruction(state, reconstruction_label)?;
    let image_ref = resolve_camera_image(state, id, selector)?;
    let index = image_ref.index();
    let reproj_error = image_error_stats(state, image_ref);

    let node = state.node(id).expect("just resolved");
    let recon = node.recon();
    let image = &recon.image_table.images[index];
    let camera = &recon.image_table.cameras[image.camera_index as usize];
    let quaternion = image.quaternion_wxyz;
    Ok(json!({
        "reconstruction_label": node.label,
        "index": index,
        "name": image.name,
        "camera_intrinsics": {
            "index": image.camera_index as usize,
            "model": camera.model.model_name(),
            "width": camera.width,
            "height": camera.height,
        },
        "quaternion_wxyz": [quaternion.w, quaternion.i, quaternion.j, quaternion.k],
        "translation_xyz": render::vector(&image.translation_xyz),
        "center": render::point(&image.camera_center()),
        "observations": render::observations_per_image(recon)[index],
        "reproj_error": reproj_error,
    }))
}

/// This image's reprojection-error summary, or `null` when the numbers are not
/// available.
///
/// The source is `compute_observation_reprojection_errors`, which is what the
/// Image Detail panel's error heatmap reads — so a figure the agent is told
/// matches the colour the human is looking at. It reads the image's `.sift`
/// file to do it, which is also why this can come back `null`: an
/// `embedded_patches` reconstruction has no `.sift` companion, and a
/// `sift_files` one whose workspace has moved cannot find it. A missing
/// summary is not a failed call — the pose and the observation count are still
/// the answer to the question that was asked.
fn image_error_stats(state: &AppState, image: ImageRef) -> Value {
    let Some(node) = state.node(image.recon) else {
        return Value::Null;
    };
    match node
        .recon()
        .compute_observation_reprojection_errors(image.index())
    {
        Ok(errors) => {
            let mut errors: Vec<f32> = errors.into_iter().map(|(_, error)| error).collect();
            render::error_stats(&mut errors)
        }
        Err(e) => {
            log::debug!(
                "MCP: no reprojection errors for {} image {}: {e}",
                node.label,
                image.index()
            );
            Value::Null
        }
    }
}

pub(super) fn get_camera_intrinsics(
    state: &mut AppState,
    reconstruction_label: Option<&str>,
    index: usize,
) -> JsonReply {
    let id = super::resolve_reconstruction(state, reconstruction_label)?;
    resolve_camera_intrinsics(state, id, index)?;
    let node = state.node(id).expect("just resolved");
    let recon = node.recon();
    let camera = &recon.image_table.cameras[index];
    let users: Vec<usize> = recon
        .image_table
        .images
        .iter()
        .enumerate()
        .filter(|(_, image)| image.camera_index as usize == index)
        .map(|(i, _)| i)
        .collect();
    let mut out = render::camera_intrinsics(camera);
    let object = out.as_object_mut().expect("camera_intrinsics is an object");
    object.insert("reconstruction_label".into(), json!(node.label));
    object.insert("index".into(), json!(index));
    object.insert("camera_image_indices".into(), json!(users));
    Ok(out)
}

pub(super) fn get_point(state: &mut AppState, query: &crate::goto_point::PointQuery) -> JsonReply {
    let point_ref = resolve_point(state, query)?;
    let point_index = point_ref.index();
    let recon_id = point_ref.recon;

    // The per-observation pixel lives in the `.sift` file for a `sift_files`
    // reconstruction, so the track is read through the same cache the Point
    // Track panel fills. Warmed before the immutable borrow below, one image at
    // a time, because `ensure_sift_cached` needs `&mut` on the cache while the
    // reconstruction it reads is borrowed out of the same `AppState`.
    warm_track_sift_cache(state, point_ref);

    let node = state
        .node(recon_id)
        .ok_or_else(|| ToolError::new("The reconstruction is no longer loaded."))?;
    let recon = node.recon();
    // The point and its track come through the overlay, so an agent is told
    // what the panels show; the images and cameras are the base's.
    let view = node.edited().point(point_ref.point).ok_or_else(|| {
        ToolError::new("That point is not in this version of the reconstruction.")
    })?;
    let point = view.point();
    let feature_indexes = view.feature_indexes();

    let track: Vec<Value> = view
        .observations()
        .iter()
        .enumerate()
        .map(|(k, observation)| {
            let image_index = observation.image_index as usize;
            let image = &recon.image_table.images[image_index];
            let camera = &recon.image_table.cameras[image.camera_index as usize];
            let xy = observation_xy(
                state,
                recon_id,
                image_index,
                feature_indexes.map(|f| f[k] as usize),
                view.keypoint_xy(k),
            );
            let (reproj_error, _) =
                crate::metrics::compute_observation_metrics(point, image, camera, xy);
            json!({
                "camera_image_index": image_index,
                "name": image.name,
                "xy": [xy[0], xy[1]],
                // NaN where the point falls behind this camera, which JSON
                // cannot carry — reported as null, which is the honest shape
                // for "this observation has no reprojection error".
                "reproj_error": reproj_error.is_finite().then_some(reproj_error),
            })
        })
        .collect();

    Ok(json!({
        "id": point_id(node, point_index),
        "reconstruction_label": node.label,
        "index": point_index,
        "position": render::point(&point.position),
        "color": point.color,
        "error": point.error,
        "at_infinity": point.is_at_infinity(),
        "track": track,
    }))
}

/// Read every `.sift` file this point's track touches into `AppState`'s cache.
///
/// A no-op for an `embedded_patches` reconstruction, whose keypoints are in the
/// file already, and for a track whose features are cached from a previous
/// call or from the panel having shown the same point.
fn warm_track_sift_cache(state: &mut AppState, point: crate::scene::PointRef) {
    let Some(node) = state.node(point.recon) else {
        return;
    };
    if node.recon().feature_indexes().is_none() {
        return;
    }
    let images: Vec<usize> = node.edited().track_image_indices(point.point);
    for image_index in images {
        let read_count = state
            .node(point.recon)
            .map(|node| node.recon().point_set.max_track_feature_index[image_index] as usize + 1)
            .unwrap_or(0);
        // Split the borrow: the cache is `&mut` while the reconstruction it
        // reads from is `&`, which is exactly why `ensure_sift_cached` is a free
        // function over the two rather than a method.
        let AppState {
            scene, sift_cache, ..
        } = state;
        let Some(node) = crate::scene::node_by_id(scene, point.recon) else {
            return;
        };
        // No progress: this warms the cache for one agent's query rather than
        // for a frame, and the tool's own entry is a `<1 ms` row that nothing
        // expands.
        ensure_sift_cached(
            sift_cache,
            node.recon(),
            ImageRef::new(point.recon, image_index),
            read_count,
            &Progress::none(),
        );
    }
}

/// The pixel one observation sits at, from whichever source this
/// reconstruction stores it in.
///
/// `(0, 0)` when the `.sift` file could not be read — the same fallback
/// `point_track_detail::prepare` uses, so the two panels and this tool agree on
/// what an unreadable feature looks like rather than each inventing an answer.
fn observation_xy(
    state: &AppState,
    recon: crate::scene::ReconId,
    image_index: usize,
    feature_index: Option<usize>,
    keypoint_xy: Option<[f32; 2]>,
) -> [f32; 2] {
    if let Some(feature) = feature_index {
        return state
            .sift_cache
            .get(&ImageRef::new(recon, image_index))
            .and_then(|sift| sift.positions_xy.get(feature))
            .copied()
            .unwrap_or([0.0, 0.0]);
    }
    keypoint_xy.unwrap_or([0.0, 0.0])
}
