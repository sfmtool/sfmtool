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

use serde_json::{json, Value};

use super::{
    render, resolve_camera_image, resolve_camera_intrinsics, resolve_point, CameraImageSel,
    JsonReply, ToolError,
};
use crate::action_log::{ActionLog, Actor};
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
) -> JsonReply {
    let log = &state.action_log;
    let limit = limit.min(ACTION_LOG_MAX_LIMIT);
    let mut matching = log
        .since(since_revision)
        .filter(|entry| actors.contains(&entry.actor));
    let entries: Vec<Value> = matching
        .by_ref()
        .take(limit)
        .map(|e| entry(log, e))
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
fn entry(log: &ActionLog, entry: &crate::action_log::Entry) -> Value {
    let mut row = json!({
        "revision": entry.revision,
        "at": log.format_rfc3339(entry.at),
        "actor": entry.actor.wire_name(),
        "kind": entry.kind.wire_name(),
        "failed": entry.failed,
        "text": entry.text,
    });
    if let crate::action_log::Kind::Query(tool) = entry.kind {
        row.as_object_mut()
            .expect("a log row is an object")
            .insert("tool".into(), json!(tool));
    }
    row
}

pub(super) fn list_camera_images(
    state: &mut AppState,
    reconstruction_label: Option<&str>,
    offset: usize,
    limit: usize,
) -> JsonReply {
    let id = super::resolve_reconstruction(state, reconstruction_label)?;
    let node = state.node(id).expect("just resolved");
    let recon = &node.recon;
    let total = recon.images.len();
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
    let recon = &node.recon;
    let image = &recon.images[index];
    let camera = &recon.cameras[image.camera_index as usize];
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
        .recon
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
    let recon = &node.recon;
    let camera = &recon.cameras[index];
    let users: Vec<usize> = recon
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
    let recon = &node.recon;
    let point = &recon.points[point_index];
    let observation_start = recon.observation_offsets[point_index];
    let feature_indexes = recon.feature_indexes();
    let keypoints_xy = recon.keypoints_xy();

    let track: Vec<Value> = recon
        .observations_for_point(point_index)
        .iter()
        .enumerate()
        .map(|(k, observation)| {
            let image_index = observation.image_index as usize;
            let image = &recon.images[image_index];
            let camera = &recon.cameras[image.camera_index as usize];
            let xy = observation_xy(
                state,
                recon_id,
                image_index,
                observation_start + k,
                feature_indexes,
                keypoints_xy,
            );
            let (reproj_error, _) = crate::point_track_detail::compute_observation_metrics(
                &point.position,
                image,
                camera,
                xy,
            );
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
        "id": point_id(recon, point_index),
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
    if node.recon.feature_indexes().is_none() {
        return;
    }
    let images: Vec<usize> = node
        .recon
        .track_image_indices(point.index())
        .into_iter()
        .collect();
    for image_index in images {
        let read_count = state
            .node(point.recon)
            .map(|node| node.recon.max_track_feature_index[image_index] as usize + 1)
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
        ensure_sift_cached(
            sift_cache,
            &node.recon,
            ImageRef::new(point.recon, image_index),
            read_count,
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
    observation: usize,
    feature_indexes: Option<&[u32]>,
    keypoints_xy: Option<&ndarray::Array2<f32>>,
) -> [f32; 2] {
    if let Some(feature_indexes) = feature_indexes {
        let feature = feature_indexes[observation] as usize;
        return state
            .sift_cache
            .get(&ImageRef::new(recon, image_index))
            .and_then(|sift| sift.positions_xy.get(feature))
            .copied()
            .unwrap_or([0.0, 0.0]);
    }
    if let Some(keypoints) = keypoints_xy {
        return [keypoints[[observation, 0]], keypoints[[observation, 1]]];
    }
    [0.0, 0.0]
}
