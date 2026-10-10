// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The tools that read and work the bench beside a node.
//!
//! See `specs/gui/bench.md` § "The wire". Every step here is one `AppState`
//! call from [`crate::bench`] -- the same call a Track View button or
//! the Image Detail menu entry makes -- so an agent's verdict, split or commit
//! is a version in the history the human is looking at, with the same Action
//! Log row and the same Undo. What this module adds is the resolution of a wire
//! handle to a node and to an item, and the two reads, which have no panel
//! gesture behind them because a panel shows what they answer.
//!
//! **A step that names no track acts on the focused item**, which is what a
//! bench panel's gesture means when it names no item: the panel resolves it
//! with [`AppState::focused_item_label`], and so does [`target`].
//!
//! The replies are [`super::edit`]'s, because a bench step is an edit of the
//! bench half of the version: [`super::edit::edited`] carries the version the
//! step pushed, `changed` for whether it pushed one at all, and the sentence it
//! recorded, and the two steps that read photographs answer with a handle when
//! they outlive [`super::REPLY_DIRECTLY_WITHIN`], exactly as `bundle_adjust`
//! does.
//!
//! **A step that had no effect answers with the version the node already stands
//! at**, `changed: false`, and its own no-effect sentence under `report` --
//! never the previous step's label, which is what a reply assembled from the
//! cursor alone would echo. The four tools that name a pixel add `clamped` and
//! the `pixel` they used, because a pixel off the photograph is brought inside
//! it rather than refused (`specs/gui/bench.md` section "The wire").

use serde_json::{json, Value};
use sfmtool_core::bench::{
    Bench, Edge, EditableTrack, Observation, Provenance, Stage, StageKind, Thresholds, Viewpoint,
};
use sfmtool_core::patch::self_similarity::{
    PatchEllipse, SelfSimilarityEllipse, SelfSimilarityEllipseUnits,
};

use super::{
    edit, resolve_camera_image, resolve_point_in, resolve_reconstruction, BackgroundReply,
    CameraImageSel, Deferred, JsonReply, Outcome, ResizeTarget, ThresholdChange, ToolError,
    TranslateTarget, VerdictAction, VerdictRows, ViewpointSel,
};
use crate::bench::{PatchEdit, Seed};
use crate::scene::ReconId;
use crate::state::AppState;

// ── The two reads ───────────────────────────────────────────────────────

/// `get_bench`: the node's bench, as the Scene tree's Bench group.
pub(super) fn get_bench(state: &AppState, label: &str) -> JsonReply {
    let id = resolve_reconstruction(state, Some(label))?;
    let bench = state
        .bench(id)
        .ok_or_else(|| ToolError::new(crate::state::NOT_LOADED))?;
    let focused = state.focused_item_label(id);
    let items: Vec<Value> = bench
        .entries()
        .iter()
        .filter_map(|entry| {
            let track = entry.item.as_track()?;
            let (kept, out) = track.verdict_counts();
            Some(json!({
                "item": entry.label,
                "kind": "track",
                "focused": focused == Some(entry.label.as_str()),
                "stage": track.stage_kind().to_string(),
                "origin": origin(track),
                "evaluation": evaluation(state, id, &entry.label),
                "counts": {
                    "observations": track.observations.len(),
                    "in": kept,
                    "out": out,
                },
            }))
        })
        .collect();
    Ok(json!({
        "reconstruction_label": node_label(state, id),
        // What an omitted `track` resolves to; null when the focused item is
        // on another node's bench or nothing is focused.
        "focused_item": focused,
        "items": items,
        // The files a search reads, reported here rather than on the track
        // because they are the node's: every track's search goes through the
        // same files.
        "index_files": index_files(state, id),
    }))
}

/// The index files beside a node, as every reply that names them states
/// them: one object per file.
///
/// `state` is the fact a caller acts on -- `current` is the one a search runs
/// against -- and `path` is the file, which is the node's own path for it even
/// when nothing is open, so an agent can see where a build would put one.
pub(super) fn index_files(state: &AppState, id: ReconId) -> Value {
    let index = state.sift_index(id);
    let patches = state.cluster_patches(id);
    let shown = |path: Option<std::path::PathBuf>| path.map(|path| path.display().to_string());
    json!({
        "sift_index": {
            "state": state.sift_index_state(id).name(),
            "path": shown(index.map(|index| index.path.clone()).or_else(|| state.sift_index_path(id))),
            "descriptors": index.map(|index| index.feature_count()),
            "images": index.map(|index| index.images),
            "stale_reason": index.and_then(|index| index.stale_reason()),
        },
        "cluster_patches": {
            "state": state.cluster_patches_state(id).name(),
            "path": shown(
                patches
                    .map(|file| file.path.clone())
                    .or_else(|| state.cluster_patches_path(id))
            ),
            "clusters": patches.map(|file| file.clusters),
            "members": patches.map(|file| file.members),
            "images": patches.map(|file| file.images),
            "stale_reason": patches.and_then(|file| file.stale_reason()),
        },
    })
}

/// `get_bench_track`: one track's table, which is Track View's Edited-mode
/// reading of it.
///
/// An observation is addressed by its position in `observations`, and that
/// position is stable across every bench step: observations are appended and
/// no step renumbers them, so an index held across a verdict or an evaluation
/// still names the same observation. Deleting an image is the one edit that
/// renumbers them.
pub(super) fn get_bench_track(state: &AppState, label: &str, named: Option<&str>) -> JsonReply {
    let (id, item) = target(state, label, named)?;
    let bench = state
        .bench(id)
        .ok_or_else(|| ToolError::new(crate::state::NOT_LOADED))?;
    let track = bench
        .track(&item)
        .ok_or_else(|| no_such_item(bench, &item))?;
    let stage = track.stage_kind();
    let (kept, out) = track.verdict_counts();
    let observations = observation_rows(state, id, track);
    Ok(json!({
        "reconstruction_label": node_label(state, id),
        "item": item,
        "kind": "track",
        "focused": state.focused_item_label(id) == Some(item.as_str()),
        // Track View's highlighted rows. Only the focused item has any.
        "selected_observations": state.selected_bench_observations(id, &item),
        "stage": stage.to_string(),
        "origin": origin(track),
        "thresholds": thresholds(&track.thresholds),
        "counts": {
            "observations": track.observations.len(),
            "in": kept,
            "out": out,
        },
        "stage_data": stage_data(track, state.node(id).map(|node| node.recon())),
        "evaluation": evaluation(state, id, &item),
        "observations": observations,
    }))
}

/// One object per observation of `track` on `id`, as `get_bench_track`
/// reports them: its image, provenance, verdict, pin, pixel and the
/// measurements of each stage, and the Jacobian and zoom of its patch in its
/// photograph, per patch-grid px at the reconstruction's patch resolution
/// (the *Zoom* column prints the zoom; the Jacobian it is read from is
/// reported as a diagnostic). `get_point`'s evaluation block reports the
/// viewed track's rows in the same shape.
pub(super) fn observation_rows(state: &AppState, id: ReconId, track: &EditableTrack) -> Vec<Value> {
    let recon = state.node(id).map(|node| node.recon());
    let world_unit = recon.and_then(|recon| recon.metadata.world_space_unit.as_deref());
    track
        .observations
        .iter()
        .enumerate()
        .map(|(index, observation)| {
            let image = crate::scene::ImageRef::new(id, observation.image as usize);
            // The Jacobian the *Zoom* column's number is read from, of the
            // warp from the patch grid at the reconstruction's patch
            // resolution to the photograph, computed without the photograph
            // exactly as the table computes it. Both are null at
            // the cluster stage, on a track with no patch yet, for an
            // observation with nothing saying where it sits, and for a patch
            // whose centre is behind the camera or outside the camera model's
            // domain; `patch_zoom` is null as well for a patch seen edge on.
            let jacobian = recon
                .and_then(|recon| crate::track_view::body::patch_jacobian(recon, track, index));
            json!({
                "observation": index,
                "camera_image": observation.image,
                "camera_image_name": state.image_name(image),
                "provenance": provenance(observation.provenance),
                "verdict": observation.verdict.to_string(),
                "pinned": observation.pinned,
                "pixel": observation_pixel(observation),
                "cluster": cluster_measurement(observation),
                "track": track_measurement(observation, world_unit),
                "patch_jacobian": jacobian.map(|jacobian| jacobian.0),
                "patch_zoom": jacobian.and_then(|jacobian| jacobian.zoom_range()),
                // The sampler the view's tile is rendered with, chosen by the
                // bench's sampler choice from the same Jacobian, and the loss
                // `L` the sampler rule compares with its threshold. Both are
                // null where the Jacobian is, and the loss is null as well
                // where it is not finite.
                "sampler": jacobian.map(|jacobian| jacobian.sampler().name()),
                "sampler_minor_axis_loss": jacobian
                    .map(|jacobian| jacobian.minor_axis_loss())
                    .filter(|loss| loss.is_finite()),
            })
        })
        .collect()
}

/// Where a track's live evaluation stands, as the two reads report it: the
/// state, the refusal or failure sentence where there is one, and the radius
/// the evaluation reads at.
///
/// `evaluating` is the answer while an evaluation of the current inputs is
/// running or waits to start, and `running` says which of the two; the numbers
/// beside it are then the previous evaluation's, and a read after the worker
/// lands says `current`.
fn evaluation(state: &AppState, id: ReconId, item: &str) -> Value {
    let evaluation = state
        .bench_evaluation(id, item)
        .unwrap_or(crate::bench::live::Evaluation::Evaluating);
    json!({
        "state": evaluation.name(),
        "reason": evaluation.reason(),
        "running": state.bench_evaluation_running(id, item),
    })
}

// ── The steps ───────────────────────────────────────────────────────────

/// `create_bench_cluster`: a cluster-stage track from a place in one camera
/// image, focused, under `named` when the call gives a label.
pub(super) fn create_bench_cluster(
    state: &mut AppState,
    label: &str,
    selector: &CameraImageSel,
    seed: &Seed,
    named: Option<&str>,
) -> JsonReply {
    let id = resolve_reconstruction(state, Some(label))?;
    view_only(state, id)?;
    let image = resolve_camera_image(state, id, selector)?;
    let mut made: Option<crate::bench::Seeded> = None;
    let reply = edit::edited(state, id, |state| {
        state.start_bench_cluster(image, seed, named).map(|seeded| {
            made = Some(seeded);
        })
    })?;
    let made = made.expect("the step reported what it made");
    let mut reply = with_item(reply, &made.label);
    insert_clamp(&mut reply, Some(made.pixel), made.clamped_from);
    Ok(reply)
}

/// `create_bench_track`: a point of the reconstruction put on the bench as a
/// track-stage track, focused.
///
/// Putting on a point a track already came from focuses that track rather
/// than putting a second one on, which is the step's own rule; the reply names
/// the item either way. A `named` other than that track's label is refused by
/// the step, naming the label the track has.
///
/// The focus pushes no version and writes only a `Selection` row, which an
/// edit reply skips, so that reply carries a `report` of its own saying what
/// this call did; without one, the only sentence in it would be the `label`
/// of the version the node stands at, which an earlier step wrote.
pub(super) fn create_bench_track(
    state: &mut AppState,
    label: &str,
    query: &crate::goto_point::PointQuery,
    named: Option<&str>,
) -> JsonReply {
    let id = resolve_reconstruction(state, Some(label))?;
    let point = resolve_point_in(state, id, query)?;
    let mut made = String::new();
    let mut reply = edit::edited(state, id, |state| {
        state.put_point_on_bench(point, named).map(|label| {
            made = label;
        })
    })?;
    if reply["changed"] == json!(false) {
        insert(
            &mut reply,
            "report",
            json!(format!(
                "Put point {} on the bench: no effect, it is on the bench already as {made}, \
                 now the focused item",
                point.point
            )),
        );
    }
    Ok(with_item(reply, &made))
}

/// `create_track_at_pixel`: a track built at a pixel of one camera image by
/// the track-at-pixel cascade, put on the bench and committed, on a worker.
///
/// Image Detail's *Create Track Here* over the wire, through the same
/// `AppState` call, so the two versions it leaves and their rows are the
/// panel's. A refusal in front of the worker (the node busy, the image not
/// posed, a `sift_files` node, a pixel off the photograph) is a tool error in
/// the call; the cascade's own refusal arrives with the task.
pub(super) fn create_track_at_pixel(
    state: &mut AppState,
    label: &str,
    selector: &CameraImageSel,
    pixel: [f64; 2],
    named: Option<&str>,
) -> Outcome {
    let id = match resolve_reconstruction(state, Some(label)) {
        Ok(id) => id,
        Err(error) => return Outcome::Done(Err(error)),
    };
    let image = match resolve_camera_image(state, id, selector) {
        Ok(image) => image,
        Err(error) => return Outcome::Done(Err(error)),
    };
    if let Err(message) = state.start_create_track_at_pixel(image, pixel, named) {
        return Outcome::Done(Err(ToolError::new(message)));
    }
    let task = state
        .background_task()
        .expect("the step started a background task");
    Outcome::Deferred(Deferred::Background(BackgroundReply {
        operation_id: task.id,
        operation_name: task.operation.name,
        answer: super::Answer::CreatedTrack(id),
        label: task.label.clone(),
        started: task.started,
    }))
}

/// What `create_track_at_pixel` answers once its run has landed.
///
/// A committed point answers as `commit_bench_track` does, with the version the
/// commit pushed, the item and the `point` it wrote, and adds the `member` that
/// built the track. Its `report` is the sentence of the row that put the track
/// on the bench. A cascade refusal is a tool error carrying the row's sentence and
/// then each member's stage and reason, one line each, in the order they were
/// tried; a commit refusal is one naming the item the track stays on the bench
/// as.
pub(super) fn created_track_reply(
    state: &AppState,
    id: ReconId,
    outcome: &Result<String, String>,
    created: Option<&crate::bench::track_at_pixel::CreatedTrack>,
) -> Result<super::ToolOutput, ToolError> {
    use crate::bench::track_at_pixel::CreatedTrack;
    match (created, outcome) {
        (
            Some(CreatedTrack::Committed {
                item,
                member,
                point,
            }),
            Ok(report),
        ) => {
            let mut reply = edit::version_reply(state, id, Some(report.clone()))?;
            insert(&mut reply, "changed", json!(true));
            let mut reply = with_item(reply, item);
            insert(&mut reply, "member", json!(member));
            insert(&mut reply, "point", point_written(state, id, *point));
            Ok(super::ToolOutput::Json(reply))
        }
        (Some(CreatedTrack::NotCommitted { why, .. }), _) => Err(ToolError::new(why.clone())),
        (Some(CreatedTrack::Refused { refusals }), Err(sentence)) => {
            let mut text = sentence.clone();
            text.push_str("\nEach member's refusal, in the order tried:");
            for line in refusals {
                text.push_str("\n- ");
                text.push_str(line);
            }
            Err(ToolError::new(text))
        }
        // A cancellation, or a run that never reached the cascade: the row's
        // own sentence.
        (_, Err(message)) => Err(ToolError::new(message.clone())),
        (_, Ok(report)) => Err(ToolError::new(format!(
            "The run ended without a track to report: {report}"
        ))),
    }
}

/// `find_nearby_tracks`: the tracks near a pixel of one camera image found,
/// put on the bench and the new ones committed, as one version, on a worker.
///
/// Image Detail's *Find Nearby Tracks* over the wire, through the same
/// `AppState` call, with `commit` and the group `label` the entry does not
/// offer. A refusal in front of the worker (the node busy, the image not
/// posed, a pixel off the photograph, and with `commit` a `sift_files` node)
/// is a tool error in the call.
pub(super) fn find_nearby_tracks(
    state: &mut AppState,
    label: &str,
    selector: &CameraImageSel,
    pixel: [f64; 2],
    commit: bool,
    named: Option<&str>,
) -> Outcome {
    let id = match resolve_reconstruction(state, Some(label)) {
        Ok(id) => id,
        Err(error) => return Outcome::Done(Err(error)),
    };
    let image = match resolve_camera_image(state, id, selector) {
        Ok(image) => image,
        Err(error) => return Outcome::Done(Err(error)),
    };
    if let Err(message) = state.start_find_nearby_tracks(image, pixel, commit, named) {
        return Outcome::Done(Err(ToolError::new(message)));
    }
    let task = state
        .background_task()
        .expect("the step started a background task");
    Outcome::Deferred(Deferred::Background(BackgroundReply {
        operation_id: task.id,
        operation_name: task.operation.name,
        answer: super::Answer::FoundNearby(id),
        label: task.label.clone(),
        started: task.started,
    }))
}

/// What `find_nearby_tracks` answers once its run has landed.
///
/// The version the node stands at, `changed` for whether the find pushed one,
/// and the row's sentence under `report`, as every edit answers; then what was
/// found: the group label, the layers, every usable track in label order with
/// what became of it, and what each source did. A distance that is infinite,
/// the far end of a far layer's range, is `null`.
pub(super) fn found_nearby_reply(
    state: &AppState,
    id: ReconId,
    outcome: &Result<String, String>,
    found: Option<&crate::bench::nearby_tracks::FoundNearby>,
) -> Result<super::ToolOutput, ToolError> {
    use crate::bench::nearby_tracks::Landing;
    use sfmtool_core::bench::FarFieldTrigger;
    let (report, found) = match (outcome, found) {
        (Ok(report), Some(found)) => (report, found),
        (Err(message), _) => return Err(ToolError::new(message.clone())),
        (Ok(report), None) => {
            return Err(ToolError::new(format!(
                "The run ended without tracks to report: {report}"
            )))
        }
    };
    let finite = |x: f64| x.is_finite().then_some(x);
    let range = |r: [f64; 2]| json!([finite(r[0]), finite(r[1])]);
    let point_id = |point: u32| {
        state
            .node(id)
            .map(|node| crate::scene::point_id(node, point as usize))
    };
    let layers = &found.found.layers;
    let ranking = |layer: usize| layers.get(layer).and_then(|l| l.ranking.as_ref());

    let mut reply = edit::version_reply(state, id, Some(report.clone()))?;
    insert(&mut reply, "changed", json!(found.changed));
    insert(&mut reply, "group_label", json!(found.found.group_label));
    insert(
        &mut reply,
        "layers",
        Value::Array(
            layers
                .iter()
                .map(|layer| {
                    json!({
                        "range": range(layer.range),
                        "rank": layer.ranking.as_ref().map(|r| r.rank),
                        "confidence": layer.ranking.as_ref().map(|r| r.confidence),
                        "members": layer.members.len(),
                        "nearest_px": layer.nearest_px,
                    })
                })
                .collect(),
        ),
    );
    let tracks: Vec<Value> = found
        .landed
        .iter()
        .map(|(k, landing)| {
            let t = &found.found.tracks[*k];
            let layer = t.layer;
            let mut row = json!({
                "label": t.label,
                "item": landing.item(),
                "point": landing.point().map(|p| json!({ "index": p, "id": point_id(p) })),
                "existing": matches!(landing, Landing::Existing { .. }),
                "committed": matches!(landing, Landing::Committed { .. }),
                "source": t.source().name(),
                "layer": layer,
                "rank": layer.and_then(ranking).map(|r| r.rank),
                "confidence": layer.and_then(ranking).map(|r| r.confidence),
                "range": range(t.range),
                "pixel": t.query_pixel(),
                "distance_px": t.distance_px(),
                "n_views": t.n_views(),
            });
            let error = match landing {
                Landing::OnBench { why: Some(why), .. } | Landing::NotBuilt { why } => Some(why),
                _ => None,
            };
            if let Some(error) = error {
                insert(&mut row, "error", json!(error));
            }
            row
        })
        .collect();
    insert(&mut reply, "tracks", Value::Array(tracks));
    let r = &found.found.report;
    insert(
        &mut reply,
        "sources",
        Value::Array(
            r.sources
                .iter()
                .map(|s| {
                    json!({
                        "source": s.source.name(),
                        "found": s.found,
                        "skipped": s.skipped,
                        "seconds": s.seconds,
                    })
                })
                .collect(),
        ),
    );
    insert(
        &mut reply,
        "stopped_after",
        json!(r.stopped_after.map(|s| s.name())),
    );
    insert(&mut reply, "duplicates", json!(r.duplicates));
    insert(
        &mut reply,
        "far_field",
        match &r.far_field {
            None => Value::Null,
            Some(run) => json!({
                "trigger": match run.trigger {
                    FarFieldTrigger::Always => "always",
                    FarFieldTrigger::NoLayer => "no_layer",
                    FarFieldTrigger::SeveralLayers => "several_layers",
                    FarFieldTrigger::NoneAtPixel => "none_at_pixel",
                },
                "found": run.report.found,
                "dropped": run.dropped,
                "seconds": run.report.seconds,
            }),
        },
    );
    Ok(super::ToolOutput::Json(reply))
}

/// `focus_bench_item`: the item Track View edits, and the item a bench tool
/// acts on when it names none.
///
/// Not a step: it pushes no version, so the reply is the item now focused and
/// `changed` for whether it was not focused before, as the selection tools
/// answer with what they left selected rather than a version.
pub(super) fn focus_bench_item(state: &mut AppState, label: &str, item: &str) -> JsonReply {
    let id = resolve_reconstruction(state, Some(label))?;
    let before = state.focused_item().copied();
    state.focus_bench_item(id, item).map_err(ToolError::new)?;
    Ok(json!({
        "reconstruction_label": node_label(state, id),
        "item": item,
        "changed": state.focused_item().copied() != before,
    }))
}

/// `unfocus_bench_item`: Track View's *Edit* box cleared. Every item stays on
/// its bench and none is focused.
///
/// Not a step, and it names no node: there is one focused item for the
/// viewer. The reply names the item it unfocused, or null for both fields and
/// `changed: false` when nothing was focused.
pub(super) fn unfocus_bench_item(state: &mut AppState) -> JsonReply {
    let before = state.focused_item().copied();
    let named = before.map(|focused| {
        (
            node_label(state, focused.node),
            state.focused_item_label(focused.node).map(str::to_string),
        )
    });
    state.unfocus_bench_item();
    let (node, item) = match named {
        Some((node, item)) => (Some(node), item),
        None => (None, None),
    };
    Ok(json!({
        "reconstruction_label": node,
        "item": item,
        "changed": before.is_some(),
    }))
}

/// `rename_bench_item`: the item under a name of the caller's own.
///
/// The reply carries the **new** label, as `save_reconstruction`'s carries a
/// renamed node's: it is the handle every later call has to use.
pub(super) fn rename_bench_item(
    state: &mut AppState,
    label: &str,
    item: &str,
    to: &str,
) -> JsonReply {
    let id = resolve_reconstruction(state, Some(label))?;
    let reply = edit::edited(state, id, |state| state.rename_bench_item(id, item, to))?;
    Ok(with_item(reply, to))
}

/// `duplicate_bench_item`: a copy of the item beside it on the bench, which the
/// reply names.
///
/// Answers as `split_bench_track` does, with the label the copy took, because
/// that is the handle every later call has to use -- and the copy is the
/// focused item, so the calls that name none already act on it. `item` omitted means
/// the focused item, as it does for the track tools.
pub(super) fn duplicate_bench_item(
    state: &mut AppState,
    label: &str,
    named: Option<&str>,
) -> JsonReply {
    let (id, item) = edit_target(state, label, named)?;
    let mut made = String::new();
    let reply = edit::edited(state, id, |state| {
        state.duplicate_bench_item(id, &item).map(|label| {
            made = label;
        })
    })?;
    let mut reply = with_item(reply, &made);
    insert(&mut reply, "copy_of", json!(item));
    Ok(reply)
}

pub(super) fn discard_bench_item(state: &mut AppState, label: &str, item: &str) -> JsonReply {
    let id = resolve_reconstruction(state, Some(label))?;
    let reply = edit::edited(state, id, |state| state.discard_bench_item(id, item))?;
    Ok(with_item(reply, item))
}

/// `add_bench_track_observation`: one more observation of the track, at the place
/// the seed names.
///
/// The reply carries the index the observation took, which is the end of the
/// list and is what the verdict and the split take back.
pub(super) fn add_bench_track_observation(
    state: &mut AppState,
    label: &str,
    named: Option<&str>,
    selector: &CameraImageSel,
    seed: &Seed,
) -> JsonReply {
    let (id, item) = edit_target(state, label, named)?;
    let image = resolve_camera_image(state, id, selector)?;
    let mut added: Option<crate::bench::Seeded> = None;
    let reply = edit::edited(state, id, |state| {
        state
            .add_bench_observation(&item, image, seed)
            .map(|seeded| {
                added = Some(seeded);
            })
    })?;
    let added = added.expect("the step reported where it seeded");
    let observation = state
        .bench_track(id, &item)
        .map(|track| track.observations.len().saturating_sub(1));
    let mut reply = with_item(reply, &item);
    insert(&mut reply, "observation", json!(observation));
    insert_clamp(&mut reply, Some(added.pixel), added.clamped_from);
    Ok(reply)
}

/// `translate_bench_patch`: the patch moved, by a displacement on its own axes
/// or to a pixel.
///
/// `by` goes to [`sfmtool_core::bench::translate_patch`]. A `pixel` with its
/// `observation` or `camera_image` goes to
/// [`sfmtool_core::bench::translate_patch_to_pixel`] as a [`Viewpoint`], and
/// the reply carries where the pointer's square landed.
pub(super) fn translate_bench_patch(
    state: &mut AppState,
    label: &str,
    named: Option<&str>,
    to: &TranslateTarget,
) -> JsonReply {
    let (id, item) = edit_target(state, label, named)?;
    match to {
        TranslateTarget::By(by) => {
            let by = *by;
            let edit = PatchEdit::Translate { by };
            let (reply, _) = patched(state, id, &item, &edit)?;
            let mut reply = with_item(reply, &item);
            insert(&mut reply, "by", json!(by));
            Ok(reply)
        }
        TranslateTarget::Pixel { viewpoint, pixel } => {
            let pixel = *pixel;
            let viewpoint = resolve_viewpoint(state, id, viewpoint)?;
            let edit = PatchEdit::TranslateToPixel { viewpoint, pixel };
            let (reply, edited) = patched(state, id, &item, &edit)?;
            let mut reply = with_item(reply, &item);
            insert_viewpoint(&mut reply, viewpoint);
            insert_clamp(
                &mut reply,
                edited.pixel.or(Some(pixel)),
                edited.clamped_from,
            );
            Ok(reply)
        }
    }
}

/// `sight_bench_observation`: one sighting put where the caller says.
///
/// **One** sighting: the cluster stage's dot, where there is no shared
/// geometry, and the track stage's dot with Track View's *Lock* cleared, which
/// is how one keypoint that settled on the wrong detail is put right. With the
/// lock ticked the track stage's dot moves the patch instead
/// (`translate_bench_patch`), because a patch every observation is a view of
/// is the thing that gesture is about. The lock is the panel's setting and the
/// wire carries no copy of it: a caller says which it means by the tool it
/// calls. Either way this writes that observation alone and pins it, because a
/// sighting a person placed is one they have ruled on, and drops the
/// measurements read at the old pixel.
pub(super) fn sight_bench_observation(
    state: &mut AppState,
    label: &str,
    named: Option<&str>,
    observation: usize,
    pixel: [f64; 2],
) -> JsonReply {
    let (id, item) = edit_target(state, label, named)?;
    let edit = PatchEdit::Sight { observation, pixel };
    let (reply, edited) = patched(state, id, &item, &edit)?;
    let mut reply = with_item(reply, &item);
    insert(&mut reply, "observation", json!(observation));
    insert_clamp(
        &mut reply,
        edited.pixel.or(Some(pixel)),
        edited.clamped_from,
    );
    Ok(reply)
}

/// `shape_bench_observation`: one cluster sighting's affine shape, set outright.
///
/// The general form of the two gestures over a parallelogram: where
/// `spin_bench_shape` turns it and `resize_bench_shape` scales it, this states
/// the whole 2x2 map from the detector's keypoint frame onto that image's
/// pixels, shear and all. The sighting is re-seeded where it is already drawn
/// and its refinement is dropped, that having been an answer about the shape it
/// was run at. A track-stage track is refused: there the shape is the patch's,
/// not the sighting's.
pub(super) fn shape_bench_observation(
    state: &mut AppState,
    label: &str,
    named: Option<&str>,
    observation: usize,
    shape: [[f64; 2]; 2],
) -> JsonReply {
    let (id, item) = edit_target(state, label, named)?;
    let edit = PatchEdit::Shape { observation, shape };
    let (reply, _) = patched(state, id, &item, &edit)?;
    let mut reply = with_item(reply, &item);
    insert(&mut reply, "observation", json!(observation));
    insert(&mut reply, "shape", json!(shape));
    Ok(reply)
}

/// `resize_bench_patch`: the track-stage patch sized, by a world half-length or
/// by putting one edge under a pixel.
///
/// `half_length` is in the reconstruction's own units and `moved_edge` says what
/// becomes of the sightings: named, that edge moves and the far one is held, so
/// the centre shifts and every sighting is carried with it; omitted, both edges
/// move about a held centre and no sighting is touched.
///
/// The pixel form is the gesture, and it is what makes the answer exact: the
/// pixel is unprojected onto the patch's own plane, so the edge really lands
/// there through whatever distortion the lens has. An `observation` says the
/// outline meant is the patch re-anchored on that sighting and the pixel is in
/// its image; a `camera_image` says it is the patch as it stands, seen in that
/// image, which is the ghost outline's edge drag.
///
/// A cluster-stage track is refused: there is no world geometry to give a world
/// half-length to, and its parallelograms are `resize_bench_shape`'s.
pub(super) fn resize_bench_patch(
    state: &mut AppState,
    label: &str,
    named: Option<&str>,
    to: &ResizeTarget,
) -> JsonReply {
    let (id, item) = edit_target(state, label, named)?;
    stage_must_be(state, id, &item, StageKind::Track, "resize_bench_shape")?;
    match to {
        ResizeTarget::HalfLength {
            half_length,
            moved_edge,
        } => {
            let (half_length, moved_edge) = (*half_length, *moved_edge);
            let edit = PatchEdit::Resize {
                half_length,
                moved_edge,
            };
            let (reply, _) = patched(state, id, &item, &edit)?;
            let mut reply = with_item(reply, &item);
            insert(&mut reply, "half_length", json!(half_length));
            insert(
                &mut reply,
                "moved_edge",
                json!(moved_edge.map(|edge| edge.name())),
            );
            Ok(reply)
        }
        ResizeTarget::Pixel {
            viewpoint,
            edge,
            pixel,
        } => {
            let (edge, pixel) = (*edge, *pixel);
            let viewpoint = resolve_viewpoint(state, id, viewpoint)?;
            let edit = PatchEdit::ResizeToPixel {
                viewpoint,
                edge,
                pixel,
            };
            let (reply, edited) = patched(state, id, &item, &edit)?;
            let mut reply = with_item(reply, &item);
            insert_viewpoint(&mut reply, viewpoint);
            insert(&mut reply, "edge", json!(edge.name()));
            insert_clamp(
                &mut reply,
                edited.pixel.or(Some(pixel)),
                edited.clamped_from,
            );
            Ok(reply)
        }
    }
}

/// `resize_bench_shape`: one cluster sighting's parallelogram sized by an edge.
///
/// The cluster stage's own half of the edge drag. There is no shared geometry
/// there, so the same arithmetic runs in that image's pixels: the shape is
/// scaled by one scalar, which keeps whatever anisotropy the detector read, and
/// the sighting moves by half the change along the dragged edge's own direction,
/// which is what holds the far edge still. Only that observation is touched. A
/// track-stage track is refused: its square is the patch's, and
/// `resize_bench_patch` is the tool for it.
pub(super) fn resize_bench_shape(
    state: &mut AppState,
    label: &str,
    named: Option<&str>,
    observation: usize,
    edge: Edge,
    pixel: [f64; 2],
) -> JsonReply {
    let (id, item) = edit_target(state, label, named)?;
    stage_must_be(state, id, &item, StageKind::Cluster, "resize_bench_patch")?;
    let edit = PatchEdit::ResizeToPixel {
        viewpoint: Viewpoint::Observation(observation),
        edge,
        pixel,
    };
    let (reply, edited) = patched(state, id, &item, &edit)?;
    let mut reply = with_item(reply, &item);
    insert(&mut reply, "observation", json!(observation));
    insert(&mut reply, "edge", json!(edge.name()));
    insert_clamp(
        &mut reply,
        edited.pixel.or(Some(pixel)),
        edited.clamped_from,
    );
    Ok(reply)
}

/// The view a pixel form named, as the core step takes it: an observation as
/// given, and a camera image resolved against the node.
fn resolve_viewpoint(
    state: &AppState,
    id: ReconId,
    viewpoint: &ViewpointSel,
) -> Result<Viewpoint, ToolError> {
    Ok(match viewpoint {
        ViewpointSel::Observation(observation) => Viewpoint::Observation(*observation),
        ViewpointSel::CameraImage(selector) => {
            Viewpoint::Image(resolve_camera_image(state, id, selector)?.image)
        }
    })
}

/// Say in the reply which view the pixel was read in, under the argument's own
/// name: `observation` or `camera_image`.
fn insert_viewpoint(reply: &mut Value, viewpoint: Viewpoint) {
    match viewpoint {
        Viewpoint::Observation(observation) => insert(reply, "observation", json!(observation)),
        Viewpoint::Image(image) => insert(reply, "camera_image", json!(image)),
    }
}

/// `tilt_bench_patch`: the patch turned to face a new outward normal.
///
/// The wire's half of the 3D viewport's arrowhead drag, and the second gesture
/// no photograph can make: a sighting says which ray the patch lies along and
/// nothing about which way the surface under it faces. The turn is the least
/// rotation onto `normal`, so no spin about the normal comes with it, and every
/// sighting keeps the in-plane offset it was measured at, rebuilt on the turned
/// axes. It stops `MAX_TILT_DEG` from any observation's camera, which is where a
/// photograph would be looking along the surface rather than at it, and the
/// sentence says which observation stopped it. A track at infinity is refused:
/// its normal is its own bearing.
///
/// The reply's `normal` is the unit normal the patch faces after the step, read
/// back off the placement, not the one named: the cap can stop the turn short
/// of it, and a caller that sends the reply's normal back gets no further turn.
pub(super) fn tilt_bench_patch(
    state: &mut AppState,
    label: &str,
    named: Option<&str>,
    normal: [f64; 3],
) -> JsonReply {
    let (id, item) = edit_target(state, label, named)?;
    let edit = PatchEdit::Tilt { normal };
    let (reply, _) = patched(state, id, &item, &edit)?;
    let applied = state
        .bench_track(id, &item)
        .and_then(|track| track.track().and_then(|payload| payload.placement.as_ref()))
        .map(|placement| {
            let n = placement.normal();
            [n.x, n.y, n.z]
        });
    let mut reply = with_item(reply, &item);
    insert(&mut reply, "normal", json!(applied));
    Ok(reply)
}

/// `spin_bench_patch`: the track-stage patch turned about its own normal.
///
/// The square turns in place, keeping its centre, its size and the face it
/// shows, so no sighting moves at all. A cluster-stage track is refused: it has
/// no patch to turn, only one affine shape per sighting, which is
/// `spin_bench_shape`'s.
pub(super) fn spin_bench_patch(
    state: &mut AppState,
    label: &str,
    named: Option<&str>,
    degrees: f64,
) -> JsonReply {
    let (id, item) = edit_target(state, label, named)?;
    stage_must_be(state, id, &item, StageKind::Track, "spin_bench_shape")?;
    let edit = PatchEdit::Spin {
        angle_rad: degrees.to_radians(),
    };
    let (reply, _) = patched(state, id, &item, &edit)?;
    let mut reply = with_item(reply, &item);
    insert(&mut reply, "degrees", json!(degrees));
    Ok(reply)
}

/// `spin_bench_shape`: one cluster sighting's parallelogram turned in its own
/// image's pixels.
///
/// Named for the part it acts on, which is why it is its own tool rather than an
/// optional observation on the patch's spin: the two turn different things by
/// different arithmetic, and a track-stage track is refused here as a cluster is
/// refused there.
pub(super) fn spin_bench_shape(
    state: &mut AppState,
    label: &str,
    named: Option<&str>,
    observation: usize,
    degrees: f64,
) -> JsonReply {
    let (id, item) = edit_target(state, label, named)?;
    stage_must_be(state, id, &item, StageKind::Cluster, "spin_bench_patch")?;
    let edit = PatchEdit::SpinShape {
        observation,
        angle_rad: degrees.to_radians(),
    };
    let (reply, _) = patched(state, id, &item, &edit)?;
    let mut reply = with_item(reply, &item);
    insert(&mut reply, "observation", json!(observation));
    insert(&mut reply, "degrees", json!(degrees));
    Ok(reply)
}

/// The refusal a tool that belongs to one stage owes an item in the other,
/// naming the tool that does belong to it.
///
/// Ahead of the step rather than through it, because the pair of tools is the
/// wire's own arrangement: core's cluster-stage edge resize is reached by the
/// same call the track stage's is, so only this surface knows that the caller
/// asked for the wrong one of two names.
fn stage_must_be(
    state: &AppState,
    id: ReconId,
    item: &str,
    wanted: StageKind,
    instead: &str,
) -> Result<(), ToolError> {
    let stage = state
        .bench_track(id, item)
        .map(|track| track.stage_kind())
        .ok_or_else(|| ToolError::new(format!("Nothing on the bench is called {item}.")))?;
    if stage == wanted {
        return Ok(());
    }
    Err(ToolError::new(format!(
        "{item} is at the {stage} stage and that is a {wanted}-stage tool; use {instead}."
    )))
}

pub(super) fn set_bench_track_verdict(
    state: &mut AppState,
    label: &str,
    named: Option<&str>,
    rows: VerdictRows,
    verdict: VerdictAction,
) -> JsonReply {
    let (id, item) = edit_target(state, label, named)?;
    let observations = match &rows {
        VerdictRows::One(observation) => vec![*observation],
        VerdictRows::Listed(listed) => listed.clone(),
        VerdictRows::All => {
            let count = state
                .bench_track(id, &item)
                .map_or(0, |track| track.observations.len());
            (0..count).collect()
        }
    };
    let reply = edit::edited(state, id, |state| match (verdict, &rows) {
        (VerdictAction::Set(verdict), VerdictRows::One(observation)) => {
            state.set_bench_verdict(id, &item, *observation, verdict)
        }
        // The parse lets several rows through only for a pin or an unpin.
        (VerdictAction::Set(_), _) => Err("Set in or out one observation at a time.".to_string()),
        (VerdictAction::Pin, _) => state.pin_bench_verdicts(id, &item, &observations),
        (VerdictAction::Unpin, _) => state.unpin_bench_verdicts(id, &item, &observations),
    })?;
    // The verdict and the pin each named observation carries now, which for
    // an unpin is what the thresholds gave it, and for a pin what it had.
    let now = |observation: usize| {
        state
            .bench_track(id, &item)
            .and_then(|track| track.observations.get(observation))
            .map(|o| (o.verdict, o.pinned))
    };
    let mut reply = with_item(reply, &item);
    match rows {
        VerdictRows::One(observation) => {
            insert(&mut reply, "observation", json!(observation));
            if let Some((verdict, pinned)) = now(observation) {
                insert(&mut reply, "verdict", json!(verdict.to_string()));
                insert(&mut reply, "pinned", json!(pinned));
            }
        }
        VerdictRows::Listed(_) | VerdictRows::All => {
            let rows: Vec<Value> = observations
                .iter()
                .filter_map(|&observation| {
                    now(observation).map(|(verdict, pinned)| {
                        json!({
                            "observation": observation,
                            "verdict": verdict.to_string(),
                            "pinned": pinned,
                        })
                    })
                })
                .collect();
            insert(&mut reply, "observations", json!(rows));
        }
    }
    Ok(reply)
}

/// `set_bench_track_reference`: Track View's *Set as reference*, one row made
/// the track's reference and pinned. The reply names the row and the reference
/// the track held before, `was`, or null: the payload's reference whether or
/// not its bitmap has been rendered yet, where `get_bench_track`'s
/// `reference_observation` is null until it is.
pub(super) fn set_bench_track_reference(
    state: &mut AppState,
    label: &str,
    named: Option<&str>,
    observation: usize,
) -> JsonReply {
    let (id, item) = edit_target(state, label, named)?;
    let was = state
        .bench_track(id, &item)
        .and_then(|track| track.track().and_then(|payload| payload.reference));
    let reply = edit::edited(state, id, |state| {
        state.set_bench_reference(id, &item, observation)
    })?;
    let mut reply = with_item(reply, &item);
    insert(&mut reply, "observation", json!(observation));
    insert(&mut reply, "was", json!(was));
    Ok(reply)
}

/// `apply_bench_track_thresholds`: the bars, and the painting they produce, as
/// one step.
///
/// One call for the two because they are one gesture: letting go of one of
/// the panel's threshold boxes applies the painting the bars produce. A bar
/// the call does not name stays where the track has it.
pub(super) fn apply_bench_track_thresholds(
    state: &mut AppState,
    label: &str,
    named: Option<&str>,
    change: &ThresholdChange,
) -> JsonReply {
    let (id, item) = edit_target(state, label, named)?;
    let bars = {
        let bench = state
            .bench(id)
            .ok_or_else(|| ToolError::new(crate::state::NOT_LOADED))?;
        let track = bench
            .track(&item)
            .ok_or_else(|| no_such_item(bench, &item))?;
        change.applied_to(&track.thresholds)
    };
    let reply = edit::edited(state, id, |state| {
        state.apply_bench_thresholds(id, &item, &bars)
    })?;
    let mut reply = with_item(reply, &item);
    insert(&mut reply, "thresholds", thresholds(&bars));
    Ok(reply)
}

/// `split_bench_track`: the named observations moved onto a second track beside
/// this one, which the reply names.
pub(super) fn split_bench_track(
    state: &mut AppState,
    label: &str,
    named: Option<&str>,
    observations: &[usize],
) -> JsonReply {
    let (id, item) = edit_target(state, label, named)?;
    let mut made = String::new();
    let reply = edit::edited(state, id, |state| {
        state
            .split_bench_track(id, &item, observations)
            .map(|label| {
                made = label;
            })
    })?;
    let mut reply = with_item(reply, &made);
    insert(&mut reply, "split_from", json!(item));
    Ok(reply)
}

/// `select_bench_observations`: Track View's row selection, replaced.
///
/// Not a step: it pushes no version, so the reply is the track and the
/// observations now selected rather than a version. The selection is the
/// bench's (`AppState::bench_rows`), so the panel highlights the rows this sets
/// and a split from the panel takes them.
pub(super) fn select_bench_observations(
    state: &mut AppState,
    label: &str,
    named: Option<&str>,
    observations: &[usize],
) -> JsonReply {
    let (id, item) = target(state, label, named)?;
    state
        .select_bench_observations(id, &item, observations)
        .map_err(ToolError::new)?;
    Ok(json!({
        "reconstruction_label": node_label(state, id),
        "item": item,
        "selected_observations": state.selected_bench_observations(id, &item),
    }))
}

/// `commit_bench_track`: the track written into the node's reconstruction.
///
/// Answers as every edit answers -- the version it pushed and the sentence the
/// Action Log recorded -- because it is one: the commit is the single bench
/// step that changes both halves of the version
/// (`specs/gui/edits/commit-track.md`).
///
/// **And it names the point it wrote**, by index and by the portable id minted
/// for it in the version that now stands, so the next call can be a `get_point`
/// rather than a search through the counts for whichever row is new. The index
/// is the reply's, not the sentence's: a commit that replaces takes the index
/// it replaced and a commit that creates takes one past the end, and neither is
/// derivable from what the Action Log row says.
///
/// **A commit onto a point that already holds exactly this track pushes no
/// version**, and answers `changed: false` with that point named as usual --
/// the same reading every other bench step's nothing-to-do gets, and what
/// stops a repeated call minting a version and an index per press.
pub(super) fn commit_bench_track(
    state: &mut AppState,
    label: &str,
    named: Option<&str>,
) -> JsonReply {
    let (id, item) = edit_target(state, label, named)?;
    let mut committed = None;
    let reply = edit::edited(state, id, |state| {
        state.commit_bench_track(id, &item).map(|written| {
            committed = Some(written);
        })
    })?;
    let written = committed.expect("a commit that succeeded named its point");
    let mut reply = with_item(reply, &item);
    insert(&mut reply, "point", point_written(state, id, written));
    Ok(reply)
}

/// The point a commit wrote, as the reply names it: the index it took, the id a
/// later call can address it by, and the index it replaced where it replaced
/// one.
///
/// The id is [`crate::scene::point_id`]'s, which is the id Track View shows
/// for the same row and the id `get_point` and `select_point` take
/// back -- a created point carries the commit's own edit hash, since there is no
/// base row to name it by.
fn point_written(state: &AppState, id: ReconId, written: crate::bench::Committed) -> Value {
    let point_id = state
        .node(id)
        .map(|node| crate::scene::point_id(node, written.point as usize));
    json!({
        "index": written.point,
        "id": point_id,
        "replaced": written.replaced,
    })
}

/// `fit_bench_track`: the track localized, re-triangulated, its bitmap re-rendered and read
/// back, on a worker thread.
pub(super) fn fit_bench_track(
    state: &mut AppState,
    label: &str,
    named: Option<&str>,
    sigma_px: Option<f64>,
) -> Outcome {
    let (id, item) = match edit_target(state, label, named) {
        Ok(target) => target,
        Err(error) => return Outcome::Done(Err(error)),
    };
    let since = state.action_log.revision();
    match state.start_bench_fit_at(id, &item, sigma_px) {
        Err(message) => Outcome::Done(Err(ToolError::new(message))),
        Ok(()) => started(state, id, since),
    }
}

/// `fit_bench_track_normal`: the track's patch turned to a normal estimated by
/// `step`, on a worker thread.
pub(super) fn fit_bench_track_normal(
    state: &mut AppState,
    label: &str,
    named: Option<&str>,
    step: crate::bench::NormalStep,
) -> Outcome {
    let (id, item) = match edit_target(state, label, named) {
        Ok(target) => target,
        Err(error) => return Outcome::Done(Err(error)),
    };
    let since = state.action_log.revision();
    match state.start_bench_normal(id, &item, step) {
        Err(message) => Outcome::Done(Err(ToolError::new(message))),
        Ok(()) => started(state, id, since),
    }
}

/// `set_bench_track_stage`: the track moved between its two representations, on
/// a worker thread.
///
/// Setting the stage a track is already at changes nothing, starts no task and
/// pushes no version; the answer is then the version the node stands at, as it
/// is for every other step that finds nothing to do.
pub(super) fn set_bench_track_stage(
    state: &mut AppState,
    label: &str,
    named: Option<&str>,
    stage: StageKind,
    sigma_px: Option<f64>,
) -> Outcome {
    let (id, item) = match edit_target(state, label, named) {
        Ok(target) => target,
        Err(error) => return Outcome::Done(Err(error)),
    };
    let since = state.action_log.revision();
    match state.start_bench_stage_at(id, &item, stage, sigma_px) {
        Err(message) => Outcome::Done(Err(ToolError::new(message))),
        Ok(()) => started(state, id, since),
    }
}

/// `search_bench_track_descriptors`: the images that hold the patch around one
/// observation, each added `out` and unpinned, on a worker thread.
pub(super) fn search_bench_track_descriptors(
    state: &mut AppState,
    label: &str,
    named: Option<&str>,
    observation: usize,
    radius_px: Option<f64>,
    min_inliers: Option<usize>,
) -> Outcome {
    let (id, item) = match edit_target(state, label, named) {
        Ok(target) => target,
        Err(error) => return Outcome::Done(Err(error)),
    };
    let since = state.action_log.revision();
    match state.start_bench_descriptor_search(id, &item, observation, radius_px, min_inliers) {
        Err(message) => Outcome::Done(Err(ToolError::new(message))),
        Ok(()) => started(state, id, since),
    }
}

/// `search_bench_track_geometry`: the photographs whose own view of the
/// track's patch matches it, each added `out` and unpinned, on a worker thread.
pub(super) fn search_bench_track_geometry(
    state: &mut AppState,
    label: &str,
    named: Option<&str>,
    observation: usize,
) -> Outcome {
    let (id, item) = match edit_target(state, label, named) {
        Ok(target) => target,
        Err(error) => return Outcome::Done(Err(error)),
    };
    let since = state.action_log.revision();
    match state.start_bench_geometry_search(id, &item, observation) {
        Err(message) => Outcome::Done(Err(ToolError::new(message))),
        Ok(()) => started(state, id, since),
    }
}

/// `open_index_files`: the node's index files, or files of the caller's
/// naming, adopted for one node.
///
/// Not an edit and not a version: the files sit beside the node's `.sfmr`,
/// and nothing about the reconstruction or the bench moves. So the reply is
/// the files rather than a version, and there is nothing for `undo` to take
/// back. A file that is not this node's opens all the same and reports
/// `state: "stale"` with the sentence saying why.
pub(super) fn open_index_files(
    state: &mut AppState,
    label: &str,
    sift_index: Option<&str>,
    cluster_patches: Option<&str>,
) -> JsonReply {
    let id = resolve_reconstruction(state, Some(label))?;
    state
        .open_index_files(
            id,
            sift_index.map(std::path::PathBuf::from),
            cluster_patches.map(std::path::PathBuf::from),
        )
        .map_err(ToolError::new)?;
    Ok(json!({
        "reconstruction_label": node_label(state, id),
        "index_files": index_files(state, id),
    }))
}

/// `close_index_files`: both files let go of, left where they are.
pub(super) fn close_index_files(state: &mut AppState, label: &str) -> JsonReply {
    let id = resolve_reconstruction(state, Some(label))?;
    state.close_index_files(id).map_err(ToolError::new)?;
    Ok(json!({
        "reconstruction_label": node_label(state, id),
        "index_files": index_files(state, id),
    }))
}

/// `build_index_files`: the node's SIFT index and its cluster patches,
/// written beside its `.sfmr` and opened, on a worker thread.
pub(super) fn build_index_files(state: &mut AppState, label: &str) -> Outcome {
    let id = match resolve_reconstruction(state, Some(label)) {
        Ok(id) => id,
        Err(error) => return Outcome::Done(Err(error)),
    };
    match state.start_build_index_files(id) {
        Err(message) => Outcome::Done(Err(ToolError::new(message))),
        Ok(()) => started_or(state, id, |state| {
            Ok(json!({
                "reconstruction_label": node_label(state, id),
                "index_files": index_files(state, id),
            }))
        }),
    }
}

/// The answer of a step that may have gone to a worker: the deferral that the
/// frame turns into a result or a handle, or the standing version where the
/// step found nothing to do and started nothing.
///
/// `since` is the Action Log revision from before the step, so the second case
/// answers with the step's own no-effect sentence rather than with nothing --
/// setting the stage a track is already at is the one step here that can find
/// nothing to do (`specs/gui/bench.md` section "The wire").
fn started(state: &AppState, id: ReconId, since: u64) -> Outcome {
    started_or(state, id, |state| edit::unchanged_reply(state, id, since))
}

/// [`started`] with the answer a step that started nothing gives.
///
/// The fallback differs by step: a bench step that found nothing to do answers
/// with the version the node stands at, and a step that touches no version
/// answers with what it is about.
fn started_or(
    state: &AppState,
    id: ReconId,
    fallback: impl FnOnce(&AppState) -> JsonReply,
) -> Outcome {
    let Some(task) = state.background_task() else {
        return super::done(fallback(state));
    };
    Outcome::Deferred(Deferred::Background(BackgroundReply {
        operation_id: task.id,
        operation_name: task.operation.name,
        answer: super::Answer::Version(id),
        label: task.label.clone(),
        started: task.started,
    }))
}

/// Where one observation of a bench track sits: the camera image it is a
/// sighting in, and the place in that image's own pixels.
///
/// What `set_image_detail_view`'s `bench_observation` target resolves to. The
/// pixel is [`crate::bench::observation_pixel`]'s, the one rule the Image
/// Detail panel's own mark and the Track View row click already share, so a
/// caller that asked to look at an observation is looking at the mark drawn for
/// it rather than at a second reading of where it is.
pub(super) fn observation_place(
    state: &AppState,
    label_of_node: ReconId,
    named: Option<&str>,
    observation: usize,
) -> Result<(crate::scene::ImageRef, [f32; 2]), ToolError> {
    let bench = state
        .bench(label_of_node)
        .ok_or_else(|| ToolError::new(crate::state::NOT_LOADED))?;
    let item = match named {
        Some(item) => item.to_string(),
        None => state
            .focused_item_label(label_of_node)
            .map(str::to_string)
            .ok_or_else(|| no_focused_item(state, label_of_node))?,
    };
    let track = bench
        .track(&item)
        .ok_or_else(|| no_such_item(bench, &item))?;
    let row = track.observations.get(observation).ok_or_else(|| {
        ToolError::new(format!(
            "{item} has {} observations; there is no observation {observation}.",
            track.observations.len()
        ))
    })?;
    let pixel = crate::bench::observation_pixel(row).ok_or_else(|| {
        ToolError::new(format!(
            "Observation {observation} of {item} has no place in its image yet -- nothing has \
             measured it and it carries no seed. sight_bench_observation places it."
        ))
    })?;
    Ok((
        crate::scene::ImageRef::new(label_of_node, row.image as usize),
        pixel,
    ))
}

// ── Resolution ──────────────────────────────────────────────────────────

/// The node a call named, and the item it acts on: the one it named, or the
/// focused item when it is on that node's bench.
///
/// A named item that is on no bench is **not** refused here. The step itself
/// refuses it, in the bench's own words, so the wire and the panel give one
/// answer to one mistake; what this resolves is only which item a call that
/// named none meant.
fn target(
    state: &AppState,
    label: &str,
    named: Option<&str>,
) -> Result<(ReconId, String), ToolError> {
    let id = resolve_reconstruction(state, Some(label))?;
    if let Some(item) = named {
        return Ok((id, item.to_string()));
    }
    let focused = state
        .focused_item_label(id)
        .map(str::to_string)
        .ok_or_else(|| no_focused_item(state, id))?;
    Ok((id, focused))
}

/// [`target`] for a tool that edits an item, refused first when the node's
/// bench is view-only (`AppState::bench_view_only_refusal`).
///
/// Asked before anything else the tool reads, so a call on a `sift_files`
/// node gets the one sentence that names the remedy, rather than a refusal
/// about a missing patch frame or a no-effect answer.
fn edit_target(
    state: &AppState,
    label: &str,
    named: Option<&str>,
) -> Result<(ReconId, String), ToolError> {
    let id = resolve_reconstruction(state, Some(label))?;
    view_only(state, id)?;
    target(state, label, named)
}

/// The view-only refusal of `id`'s bench as a tool error.
fn view_only(state: &AppState, id: ReconId) -> Result<(), ToolError> {
    match state.bench_view_only_refusal(id) {
        Some(why) => Err(ToolError::new(why)),
        None => Ok(()),
    }
}

/// The refusal a call that names no track gets when no item on the node's
/// bench is focused.
fn no_focused_item(state: &AppState, id: ReconId) -> ToolError {
    ToolError::new(format!(
        "No item on {}'s bench is focused. Name one with track, focus one with \
         focus_bench_item, or put one on with create_bench_track or create_bench_cluster.",
        node_label(state, id)
    ))
}

/// The refusal a label that names nothing gets, in the bench's own words.
fn no_such_item(bench: &Bench, item: &str) -> ToolError {
    let labels: Vec<String> = bench.labels().map(|label| format!("{label:?}")).collect();
    let on_it = if labels.is_empty() {
        "nothing is on it".to_string()
    } else {
        format!("on it: {}", labels.join(", "))
    };
    ToolError::new(format!("Nothing on the bench is called {item}; {on_it}."))
}

/// The node's label, for a message or a reply that names it.
fn node_label(state: &AppState, id: ReconId) -> String {
    state
        .node(id)
        .map(|node| node.label.clone())
        .unwrap_or_default()
}

// ── The shapes a bench reply is built from ──────────────────────────────

/// An edit's reply with the item it acted on named in it.
fn with_item(mut reply: Value, item: &str) -> Value {
    insert(&mut reply, "item", json!(item));
    reply
}

/// One patch edit, with what it did beside the version reply.
///
/// The patch tools that name a pixel all want the same two things out of the
/// step -- the pixel it used and whether it had to bring that pixel inside the
/// photograph -- and the `AppState` method is the only place either is known.
fn patched(
    state: &mut AppState,
    id: ReconId,
    item: &str,
    edit: &PatchEdit,
) -> Result<(Value, crate::bench::PatchEdited), ToolError> {
    let mut done: Option<crate::bench::PatchEdited> = None;
    let reply = edit::edited(state, id, |state| {
        state.edit_bench_patch(id, item, edit).map(|edited| {
            done = Some(edited);
        })
    })?;
    Ok((reply, done.expect("the step reported what it did")))
}

/// The pixel a step used, and whether it is the one that was asked for.
///
/// `clamped` is the fact an agent acts on -- the call named a place off the
/// photograph and the step took the nearest place on it -- and `pixel` is what
/// it took, so a reader that ignores the flag still has the right number.
fn insert_clamp(reply: &mut Value, pixel: Option<[f64; 2]>, clamped_from: Option<[f64; 2]>) {
    insert(reply, "pixel", json!(pixel));
    insert(reply, "clamped", json!(clamped_from.is_some()));
    insert(reply, "clamped_from", json!(clamped_from));
}

fn insert(reply: &mut Value, key: &str, value: Value) {
    reply
        .as_object_mut()
        .expect("a version reply is an object")
        .insert(key.to_string(), value);
}

/// The point a track was put on the bench from, by the index it held in the
/// version it was read out of.
fn origin(track: &EditableTrack) -> Value {
    match track.origin {
        Some(origin) => json!({ "point": origin.point, "version": origin.version }),
        None => Value::Null,
    }
}

pub(super) fn thresholds(bars: &Thresholds) -> Value {
    json!({
        "min_zncc": bars.min_zncc,
        "min_zncc_middle": bars.min_zncc_middle,
        "cluster_min_zncc": bars.cluster_min_zncc,
        "cluster_min_zncc_middle": bars.cluster_min_zncc_middle,
        "max_shift_px": bars.max_shift_px,
        "max_zncc_self_similarity_radius": bars.max_zncc_self_similarity_radius,
        "max_projection_error_px": bars.max_projection_error_px,
        "geometry_search_min_relative_zncc": bars.geometry_search_min_relative_zncc,
    })
}

/// What the stage the track is in carries, beside the observations: the
/// template's cut at the cluster stage, the patch at the track stage.
///
/// The template's samples and the patch bitmap are reported as present or
/// absent rather than sent: they are pictures, and this surface is not a data
/// channel. At the track stage `patch_resolution` is the patch-grid
/// resolution `R` of `recon` ([`crate::bench::patch_resolution`]), the grid
/// every observation's `patch_jacobian`, `patch_zoom`, shift and
/// self-similarity grid px are in.
fn stage_data(track: &EditableTrack, recon: Option<&sfmtool_core::SfmrReconstruction>) -> Value {
    match &track.stage {
        Stage::Cluster(payload) => json!({
            "reference": payload.reference,
            "radius": payload.radius,
            "template_cut": payload.template.is_some(),
        }),
        // A bearing and a position are the same three numbers under different
        // rules, so the coordinate is published under the name of whichever it
        // is -- `direction` for a `w = 0` track, `position` otherwise, the other
        // null -- with `at_infinity` beside them for a reader that wants the
        // flag rather than the key.
        //
        // The track's own flag, not its patch's `w`: a point put on the bench
        // from a node that stores no patch frames carries no patch at all, and
        // a bearing it came from is still a bearing.
        Stage::Track(payload) => {
            let at_infinity = payload.at_infinity;
            let coordinate = payload.position.map(|p| [p.x, p.y, p.z]);
            json!({
                "at_infinity": at_infinity,
                "position": (!at_infinity).then_some(coordinate).flatten(),
                "direction": at_infinity.then_some(coordinate).flatten(),
                "condition_number": payload.condition_number,
                "color": payload.color,
                "normal_confidence": payload.normal_confidence,
                "placement": super::render::placement(payload.placement.as_ref()),
                "has_bitmap": payload.bitmap.is_some(),
                // Whether that bitmap is one for judging only: rendered from
                // every row with a keypoint because fewer than two `in` rows
                // carry one. It names no row and a commit does not write it.
                "bitmap_for_judging": payload.bitmap_for_judging,
                // Whether that bitmap is kept only until the next render
                // replaces it: an unpin handed the reference to the rule,
                // whose pick is another row, so no row is scored against it.
                "bitmap_pending": track.bitmap_pending(),
                // The reference in use: the row the stored bitmap is rendered
                // from, as an index into `observations`, or null for a track
                // with no bitmap or one whose bitmap is the render of no row,
                // a fused mean, which the first evaluation that reads a pick
                // the rule reached other than through its last fallback
                // renders from that pick. It can differ from the rule's pick below while
                // its row is pinned, between an unpin and the render that moves
                // it to the pick, and where the pick alternated between rows
                // within one evaluation.
                "reference_observation": crate::bench::reference_in_use(track),
                "patch_resolution": recon.map(crate::bench::patch_resolution),
                // The row the reference-view rule picked at the last
                // evaluation, as an index into `observations`, or null.
                "reference_view_observation": crate::bench::reference_view_pick(track),
            })
        }
    }
}

/// What put an observation on the track, in the words the panel's *From* column
/// uses, with the number that goes with it where there is one.
fn provenance(provenance: Provenance) -> Value {
    match provenance {
        Provenance::Origin => json!({ "kind": "origin" }),
        Provenance::Descriptor { feature } => json!({ "kind": "descriptor", "feature": feature }),
        Provenance::Search { inliers } => json!({ "kind": "search", "inliers": inliers }),
        Provenance::Sweep => json!({ "kind": "sweep" }),
        Provenance::Pixel => json!({ "kind": "pixel" }),
        Provenance::Point { point } => json!({ "kind": "point", "point": point }),
    }
}

/// Where the observation sits, whatever said so.
///
/// [`crate::bench::observation_site`]'s rule, which is the panel's: the
/// keypoint the track stage carries, else the refined cluster position, else the seed
/// the step that proposed it left. It is a top level field rather than
/// something a caller assembles out of the two measurement blocks below,
/// because every reader of this surface wants the one answer those blocks are
/// read for -- an observation of a cluster has a seed and no keypoint, one of a
/// track has a keypoint, and an agent should not have to know which slot to
/// fall back to before it can look at one.
fn observation_pixel(observation: &Observation) -> Value {
    match crate::bench::observation_site(observation) {
        Some(site) => json!(site.pixel),
        None => Value::Null,
    }
}

/// The cluster stage's slot: the seed it was put there with, and what the
/// refinement made of it.
fn cluster_measurement(observation: &Observation) -> Value {
    let Some(measured) = observation.cluster.as_ref() else {
        return Value::Null;
    };
    json!({
        "seed_pixel": measured.seed_position,
        "seed_shape": measured.seed_shape,
        "pixel": measured.position,
        "shape": measured.shape,
        "zncc": finite(measured.zncc),
        // The same samples read over the middle of the patch only.
        "zncc_middle": finite(measured.zncc_middle),
        // And over each ninth of the patch, rows from the top.
        "zncc_grid": grid(measured.zncc_grid),
        "shift_px": finite(measured.shift_px),
        // How far the tile can slide over itself and still match
        // itself, in grid px, 3 meaning 3 or more; over the middle and each
        // ninth too, and the tile's ZNCC against itself at every shift of the
        // 7 x 7 square.
        "zncc_self_similarity_radius": finite(measured.zncc_self_similarity_radius),
        "zncc_self_similarity_radius_middle": finite(measured.zncc_self_similarity_radius_middle),
        "zncc_self_similarity_radius_grid": grid(measured.zncc_self_similarity_radius_grid),
        "zncc_self_similarity_surface": surface(measured.zncc_self_similarity_surface.as_deref()),
        // The deficit the core was judged by: the region of matching shifts
        // is the surface at or above 1 - tolerance.
        "zncc_self_similarity_tolerance": finite(measured.zncc_self_similarity_tolerance),
        // The ellipse of that region, whose semi-major axis is the radius,
        // whole and middle: in grid px, in image px, and along the patch's u
        // and v; and each ninth's, in grid px.
        "zncc_self_similarity_ellipse": ellipse_units(measured.zncc_self_similarity_ellipse, None),
        "zncc_self_similarity_ellipse_middle": ellipse_units(measured.zncc_self_similarity_ellipse_middle, None),
        "zncc_self_similarity_ellipse_grid": ellipse_grid(measured.zncc_self_similarity_ellipse_grid),
        "status": measured.status.map(|status| format!("{status:?}")),
    })
}

/// The track stage's slot: where the observation sits, how well it agrees with
/// the rest, and -- when it could not be read at all -- why.
///
/// The two distances answer different questions and are both reported.
/// `seed_shift_px` is how far the correlation peak sits from the observation
/// itself, which is the sighting's own evidence; `projection_offset_px` is how
/// far the observation sits from the point's projection, which is a statement
/// about the point. A track whose position is wrong shows large offsets beside
/// zero shifts.
fn track_measurement(observation: &Observation, world_unit: Option<&str>) -> Value {
    let Some(measured) = observation.track.as_ref() else {
        return Value::Null;
    };
    json!({
        "keypoint": measured.keypoint,
        // The row's score against the stored patch bitmap, plain: 1 on the
        // row the bitmap is rendered from, null with `reason` where there is
        // no bitmap or the pair could not be read. Shown beside the
        // blur-matched score, which the bars judge.
        "plain_zncc": finite(measured.plain_zncc),
        // The same pair read over the middle of the tile only.
        "plain_zncc_middle": finite(measured.plain_zncc_middle),
        // And over each ninth of the tile, rows from the top.
        "plain_zncc_grid": grid(measured.plain_zncc_grid),
        // The same three readings with the bitmap alone blurred to the row's
        // sharpness (equal to the plain ones where it is not blurred): the
        // readings the bars judge. Then the blur's width in grid px (0 when
        // read plain), and whether the row is sharper than the bitmap.
        "blur_matched_zncc": finite(measured.blur_matched_zncc),
        "blur_matched_zncc_middle": finite(measured.blur_matched_zncc_middle),
        "blur_matched_zncc_grid": grid(measured.blur_matched_zncc_grid),
        "bitmap_blur_sigma": finite(measured.bitmap_blur_sigma),
        "sharper_than_bitmap": measured.sharper_than_bitmap,
        // How far the peak of the row's correlation with the reference's
        // render sits from the row's keypoint, in grid px: 0 on the
        // reference's own row, null with `reason` where the localizer could
        // not read the row.
        "seed_shift_px": finite(measured.seed_shift_px),
        "projection_offset_px": finite(measured.projection_offset_px),
        "reprojection_error": finite(measured.reprojection_error),
        "ray_angle_deg": finite(measured.ray_angle_deg),
        // How far the tile can slide over itself and still match
        // itself, in grid px, 3 meaning 3 or more; over the middle and each
        // ninth too, and the tile's ZNCC against itself at every shift of the
        // 7 x 7 square.
        "zncc_self_similarity_radius": finite(measured.zncc_self_similarity_radius),
        "zncc_self_similarity_radius_middle": finite(measured.zncc_self_similarity_radius_middle),
        "zncc_self_similarity_radius_grid": grid(measured.zncc_self_similarity_radius_grid),
        "zncc_self_similarity_surface": surface(measured.zncc_self_similarity_surface.as_deref()),
        // The deficit the core was judged by: the region of matching shifts
        // is the surface at or above 1 - tolerance.
        "zncc_self_similarity_tolerance": finite(measured.zncc_self_similarity_tolerance),
        // The ellipse of that region, whose semi-major axis is the radius,
        // whole and middle: in grid px, in image px, and along the patch's u
        // and v; and each ninth's, in grid px.
        "zncc_self_similarity_ellipse": ellipse_units(measured.zncc_self_similarity_ellipse, world_unit),
        "zncc_self_similarity_ellipse_middle": ellipse_units(measured.zncc_self_similarity_ellipse_middle, world_unit),
        "zncc_self_similarity_ellipse_grid": ellipse_grid(measured.zncc_self_similarity_ellipse_grid),
        // The reference view's per-view readings: the angle the view sees the
        // patch at and the direction in the patch's plane its ray leans, the
        // share of the tile with data and of the photograph under it that is
        // clipped, and, for an `in` row, its median ZNCC with the other `in`
        // rows, over the whole tile and per ninth, and its cell deficit.
        "viewing_angle_deg": finite(measured.viewing_angle_deg),
        "tilt_direction_deg": finite(measured.tilt_direction_deg),
        // The tile's `[least, most]` zoom, grid px per photograph px.
        "zoom": measured.zoom.map(|z| z.map(|v| v.is_finite().then_some(v))),
        "coverage": finite(measured.coverage),
        "clipped_share": finite(measured.clipped_share),
        "pair_zncc": finite(measured.pair_zncc),
        "pair_zncc_grid": grid(measured.pair_zncc_grid),
        "cell_deficit": finite(measured.cell_deficit),
        // What the reference-view rule decided about an `in` row.
        "reference_view": measured.reference_view.map(|standing| json!({
            "is_reference": standing.is_reference(),
            "rejected_by": standing.rejected_by.map(|test| test.name()),
            "fallback": standing.fallback.name(),
        })),
        // Present only when the last fit refused the walk and left this sighting
        // at its seed: how far the correlation peak sat, the pixel it sat at
        // and the scores of the tile there against the stored bitmap, plain
        // and blur-matched (the one judged), whole, middle and per ninth, to
        // set beside the row's own. Accepting the walk is `sight_bench_observation` with
        // `walked_to` as its pixel.
        "walked_px": finite(measured.walked_px),
        "walked_to": measured.walked_to,
        "walked_plain_zncc": finite(measured.walked_plain_zncc),
        "walked_plain_zncc_middle": finite(measured.walked_plain_zncc_middle),
        "walked_plain_zncc_grid": grid(measured.walked_plain_zncc_grid),
        "walked_blur_matched_zncc": finite(measured.walked_blur_matched_zncc),
        "walked_blur_matched_zncc_middle": finite(measured.walked_blur_matched_zncc_middle),
        "walked_blur_matched_zncc_grid": grid(measured.walked_blur_matched_zncc_grid),
        "reason": measured.reason.map(|reason| reason.to_string()),
    })
}

/// A measurement JSON can carry, or null.
///
/// A kernel can leave a NaN where it could not measure, and JSON has no NaN;
/// null is the honest shape for "this observation has no such number", which is
/// what the panel's own `-` says.
fn finite(value: Option<f64>) -> Option<f64> {
    value.filter(|v| v.is_finite())
}

/// A three-by-three grid as three rows of three, top row first, with null in a
/// cell that has no reading, as [`finite`] has it; or null for no grid.
fn grid(value: Option<[[f64; 3]; 3]>) -> Option<[[Option<f64>; 3]; 3]> {
    value.map(|rows| rows.map(|row| row.map(|v| finite(Some(v)))))
}

/// One self-similarity ellipse, or null for one with no reading: `axes`
/// `[semi-major, semi-minor]`, `axes_is_at_least` per axis, true where the
/// true length may be larger (the region at the level runs off the square of
/// shifts searched, a gap with no reading beside it could hide more of it, or
/// the length reached the largest radius searched), `major_angle` the major
/// axis's angle in radians in `[0, π)` from the frame's first axis towards its
/// second (null for a circle), and `matrix` the ellipse's 2×2 matrix `E`,
/// `dᵀ E⁻¹ d = 1` on its boundary.
fn ellipse(value: &SelfSimilarityEllipse) -> Value {
    if !value.axes.iter().all(|a| a.is_finite()) {
        return Value::Null;
    }
    json!({
        "axes": value.axes,
        "axes_is_at_least": value.axes_is_at_least,
        "major_angle": finite(Some(value.major_angle)),
        "matrix": value.matrix,
    })
}

/// A self-similarity reading's ellipse in each unit, or null for none:
/// `grid_px` in patch-grid px (`x` right, `y` down), its semi-major axis the
/// radius; `image_px` in the photograph's px (`x` right, `y` down), null where
/// the tile's centre does not project; and `patch` along the patch's `u` and
/// `v` (angle from `u` towards `v`), as `{ "kind", "unit", "ellipse" }` with
/// `kind` `"length"` and `unit` the reconstruction's `world_space_unit`
/// (`world_unit`; null for scene units), or `kind` `"angle"` and `unit`
/// `"degrees"` for a patch at infinity. `patch` is null at the cluster stage,
/// which has no patch.
fn ellipse_units(value: Option<SelfSimilarityEllipseUnits>, world_unit: Option<&str>) -> Value {
    let Some(units) = value else {
        return Value::Null;
    };
    let patch = match units.patch {
        None => Value::Null,
        Some(on_patch) => {
            let (kind, unit) = match on_patch {
                PatchEllipse::Length(_) => ("length", world_unit),
                PatchEllipse::Angle(_) => ("angle", Some("degrees")),
            };
            json!({
                "kind": kind,
                "unit": unit,
                "ellipse": ellipse(on_patch.ellipse()),
            })
        }
    };
    json!({
        "grid_px": ellipse(&units.grid_px),
        "image_px": units.image_px.as_ref().map_or(Value::Null, ellipse),
        "patch": patch,
    })
}

/// Each ninth's self-similarity ellipse in grid px as three rows of three,
/// top row first, as [`ellipse`] writes each; or null for none.
fn ellipse_grid(value: Option<[[SelfSimilarityEllipse; 3]; 3]>) -> Value {
    match value {
        None => Value::Null,
        Some(rows) => Value::Array(
            rows.iter()
                .map(|row| Value::Array(row.iter().map(ellipse).collect()))
                .collect(),
        ),
    }
}

/// A square ZNCC surface, stored row-major, as rows of numbers from the top
/// row (`dy = -r`), with null for a shift with no reading;
/// or null for no surface.
fn surface(value: Option<&[f64]>) -> Option<Vec<Vec<Option<f64>>>> {
    let values = value?;
    let side = (values.len() as f64).sqrt().round() as usize;
    if side == 0 || side * side != values.len() {
        return None;
    }
    Some(
        values
            .chunks(side)
            .map(|row| row.iter().map(|&v| finite(Some(v))).collect())
            .collect(),
    )
}
