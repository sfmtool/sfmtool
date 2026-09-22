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
//! **A step that names no track acts on the active one**, which is what a bench
//! panel's gesture means when it names no item: the panel resolves it with
//! [`crate::bench::active_track_label`], and so does [`target`].
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
    Bench, Edge, EditableTrack, Observation, Provenance, Stage, StageKind, Thresholds, Verdict,
};

use super::{
    edit, resolve_camera_image, resolve_point_in, resolve_reconstruction, BackgroundReply,
    CameraImageSel, Deferred, JsonReply, Outcome, ResizeTarget, ThresholdChange, ToolError,
    TranslateTarget,
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
        .ok_or_else(|| ToolError::new("That reconstruction is no longer loaded."))?;
    let active = crate::bench::active_track_label(bench);
    let items: Vec<Value> = bench
        .entries()
        .iter()
        .filter_map(|entry| {
            let track = entry.item.as_track()?;
            let (kept, candidates, out) = track.verdict_counts();
            Some(json!({
                "item": entry.label,
                "kind": "track",
                "active": active == Some(entry.label.as_str()),
                "stage": track.stage_kind().to_string(),
                "origin": origin(track),
                "counts": {
                    "observations": track.observations.len(),
                    "in": kept,
                    "candidate": candidates,
                    "out": out,
                },
            }))
        })
        .collect();
    Ok(json!({
        "reconstruction_label": node_label(state, id),
        // One active item per kind, which is what an omitted `track` resolves
        // to; null when none of that kind is active, which a bench holding
        // items can be once Track View's Edit box is cleared.
        "active": { "track": active },
        "items": items,
        // The index a descriptor search would query, reported here rather than
        // on the track because it is the node's: every track's search goes
        // through the same file.
        "sift_index": sift_index(state, id),
    }))
}

/// The SIFT index beside a node, as every reply that names one states it.
///
/// `state` is the fact a caller acts on -- `current` is the one a search runs
/// against -- and `path` is the file, which is the node's own index path even
/// when nothing is open, so an agent can see where a build would put one.
pub(super) fn sift_index(state: &AppState, id: ReconId) -> Value {
    let index = state.sift_index(id);
    json!({
        "state": state.sift_index_state(id).name(),
        "path": index
            .map(|index| index.path.display().to_string())
            .or_else(|| state.sift_index_path(id).map(|path| path.display().to_string())),
        "descriptors": index.map(|index| index.feature_count()),
        "images": index.map(|index| index.images),
        "stale_reason": index.and_then(|index| index.stale_reason()),
    })
}

/// `get_bench_track`: one track's table, which is Track View's edit-mode
/// reading of it.
///
/// An observation is addressed by its position in `observations`, and that
/// position is stable for the life of the track: observations are appended and
/// never renumbered, so an index held across a verdict or an evaluation still
/// names the same observation.
pub(super) fn get_bench_track(state: &AppState, label: &str, named: Option<&str>) -> JsonReply {
    let (id, item) = target(state, label, named)?;
    let bench = state
        .bench(id)
        .ok_or_else(|| ToolError::new("That reconstruction is no longer loaded."))?;
    let track = bench
        .track(&item)
        .ok_or_else(|| no_such_item(bench, &item))?;
    let stage = track.stage_kind();
    let (kept, candidates, out) = track.verdict_counts();
    let observations: Vec<Value> = track
        .observations
        .iter()
        .enumerate()
        .map(|(index, observation)| {
            let image = crate::scene::ImageRef::new(id, observation.image as usize);
            json!({
                "observation": index,
                "camera_image": observation.image,
                "camera_image_name": state.image_name(image),
                "provenance": provenance(observation.provenance),
                "verdict": observation.verdict.to_string(),
                "pinned": observation.pinned,
                "pixel": observation_pixel(observation),
                "cluster": cluster_measurement(observation),
                "track": track_measurement(observation),
            })
        })
        .collect();
    Ok(json!({
        "reconstruction_label": node_label(state, id),
        "item": item,
        "kind": "track",
        "active": crate::bench::active_track_label(bench) == Some(item.as_str()),
        "stage": stage.to_string(),
        "origin": origin(track),
        "thresholds": thresholds(&track.thresholds),
        "counts": {
            "observations": track.observations.len(),
            "in": kept,
            "candidate": candidates,
            "out": out,
        },
        "stage_data": stage_data(track),
        "observations": observations,
    }))
}

// ── The steps ───────────────────────────────────────────────────────────

/// `create_bench_cluster`: a cluster-stage track from a place in one camera
/// image, made the active one.
pub(super) fn create_bench_cluster(
    state: &mut AppState,
    label: &str,
    selector: &CameraImageSel,
    seed: &Seed,
) -> JsonReply {
    let id = resolve_reconstruction(state, Some(label))?;
    let image = resolve_camera_image(state, id, selector)?;
    let mut made: Option<crate::bench::Seeded> = None;
    let reply = edit::edited(state, id, |state| {
        state.start_bench_cluster(image, seed).map(|seeded| {
            made = Some(seeded);
        })
    })?;
    let made = made.expect("the step reported what it made");
    let mut reply = with_item(reply, &made.label);
    insert_clamp(&mut reply, Some(made.pixel), made.clamped_from);
    Ok(reply)
}

/// `create_bench_track`: a point of the reconstruction put on the bench as a
/// track-stage track, made the active one.
///
/// Putting on a point a track already came from activates that track rather
/// than putting a second one on, which is the step's own rule; the reply names
/// the item either way.
pub(super) fn create_bench_track(
    state: &mut AppState,
    label: &str,
    query: &crate::goto_point::PointQuery,
) -> JsonReply {
    let id = resolve_reconstruction(state, Some(label))?;
    let point = resolve_point_in(state, id, query)?;
    let mut made = String::new();
    let reply = edit::edited(state, id, |state| {
        state.put_point_on_bench(point).map(|label| {
            made = label;
        })
    })?;
    Ok(with_item(reply, &made))
}

pub(super) fn activate_bench_item(state: &mut AppState, label: &str, item: &str) -> JsonReply {
    let id = resolve_reconstruction(state, Some(label))?;
    let reply = edit::edited(state, id, |state| state.activate_bench_item(id, item))?;
    Ok(with_item(reply, item))
}

/// `deactivate_bench_item`: Track View's *Edit* box cleared. Every item stays
/// on the bench and none is active; with nothing active it is a no-effect
/// reply, as every bench step's nothing-to-do is.
pub(super) fn deactivate_bench_item(state: &mut AppState, label: &str) -> JsonReply {
    let id = resolve_reconstruction(state, Some(label))?;
    edit::edited(state, id, |state| state.deactivate_bench_item(id))
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
/// that is the handle every later call has to use -- and the copy is the active
/// track, so the calls that name none already act on it. `item` omitted means
/// the active track, as it does for the track tools.
pub(super) fn duplicate_bench_item(
    state: &mut AppState,
    label: &str,
    named: Option<&str>,
) -> JsonReply {
    let (id, item) = target(state, label, named)?;
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

/// `add_bench_track_observation`: one more candidate of the track, at the place
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
    let (id, item) = target(state, label, named)?;
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
/// **One tool for the two ways to say where it goes**, because they are one
/// step: a track-stage track has one patch and every observation is a view of
/// it, so both forms move the patch and not a sighting, and every keypoint is
/// carried by the same displacement, keeping its own offset from the centre's
/// projection.
///
/// `by` is `[u, v, n]` on the patch's own orthonormal axes, in world units. Its
/// tangential part slides the patch across its own plane, which a photograph can
/// also say; its normal part is the one no photograph can, a sighting naming the
/// ray the patch lies along and not how far down it the surface is. A track at
/// infinity refuses a `by` with a normal part and carries a tangential one.
///
/// The pixel form names `observation` -- the image the pixel is in, and the
/// outline the pointer is read against -- and lands the patch's centre under it.
pub(super) fn translate_bench_patch(
    state: &mut AppState,
    label: &str,
    named: Option<&str>,
    to: &TranslateTarget,
) -> JsonReply {
    let (id, item) = target(state, label, named)?;
    match *to {
        TranslateTarget::By(by) => {
            let edit = PatchEdit::Translate { by };
            let (reply, _) = patched(state, id, &item, &edit)?;
            let mut reply = with_item(reply, &item);
            insert(&mut reply, "by", json!(by));
            Ok(reply)
        }
        TranslateTarget::Pixel { observation, pixel } => {
            let edit = PatchEdit::TranslateToPixel { observation, pixel };
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
    }
}

/// `sight_bench_observation`: one sighting put where the caller says.
///
/// **One** sighting, which at the track stage is the step the panel's dot no
/// longer makes: dragging the dot there moves the patch
/// (`translate_bench_patch`), because a patch every observation is a view of is
/// the thing that gesture is about. This is what remains for a caller that
/// really means one keypoint -- the cluster stage's dot, where there is no
/// shared geometry, and a script placing one sighting of a track-stage track by
/// hand. Either way it writes that observation alone and pins it, because a
/// sighting a person placed is one they have ruled on, and drops the
/// measurements read at the old pixel.
pub(super) fn sight_bench_observation(
    state: &mut AppState,
    label: &str,
    named: Option<&str>,
    observation: usize,
    pixel: [f64; 2],
) -> JsonReply {
    let (id, item) = target(state, label, named)?;
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
    let (id, item) = target(state, label, named)?;
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
/// there through whatever distortion the lens has. The observation says whose
/// outline is meant -- the patch re-anchored on that sighting -- and the pixel is
/// in that observation's image.
///
/// A cluster-stage track is refused: there is no world geometry to give a world
/// half-length to, and its parallelograms are `resize_bench_shape`'s.
pub(super) fn resize_bench_patch(
    state: &mut AppState,
    label: &str,
    named: Option<&str>,
    to: &ResizeTarget,
) -> JsonReply {
    let (id, item) = target(state, label, named)?;
    stage_must_be(state, id, &item, StageKind::Track, "resize_bench_shape")?;
    match *to {
        ResizeTarget::HalfLength {
            half_length,
            moved_edge,
        } => {
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
            observation,
            edge,
            pixel,
        } => {
            let edit = PatchEdit::ResizeToPixel {
                observation,
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
    let (id, item) = target(state, label, named)?;
    stage_must_be(state, id, &item, StageKind::Cluster, "resize_bench_patch")?;
    let edit = PatchEdit::ResizeToPixel {
        observation,
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
pub(super) fn tilt_bench_patch(
    state: &mut AppState,
    label: &str,
    named: Option<&str>,
    normal: [f64; 3],
) -> JsonReply {
    let (id, item) = target(state, label, named)?;
    let edit = PatchEdit::Tilt { normal };
    let (reply, _) = patched(state, id, &item, &edit)?;
    let mut reply = with_item(reply, &item);
    insert(&mut reply, "normal", json!(normal));
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
    let (id, item) = target(state, label, named)?;
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
    let (id, item) = target(state, label, named)?;
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
    observation: usize,
    verdict: Verdict,
) -> JsonReply {
    let (id, item) = target(state, label, named)?;
    let reply = edit::edited(state, id, |state| {
        state.set_bench_verdict(id, &item, observation, verdict)
    })?;
    let mut reply = with_item(reply, &item);
    insert(&mut reply, "observation", json!(observation));
    insert(&mut reply, "verdict", json!(verdict.to_string()));
    Ok(reply)
}

/// `apply_bench_track_thresholds`: the bars, and the painting they produce, as
/// one step.
///
/// One call for the two because they are one gesture: the panel's sliders
/// propose until its button is pressed, and what the button applies is the
/// painting those positions produce. A bar the call does not name stays where
/// the track has it.
pub(super) fn apply_bench_track_thresholds(
    state: &mut AppState,
    label: &str,
    named: Option<&str>,
    change: &ThresholdChange,
) -> JsonReply {
    let (id, item) = target(state, label, named)?;
    let bars = {
        let bench = state
            .bench(id)
            .ok_or_else(|| ToolError::new("That reconstruction is no longer loaded."))?;
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
    let (id, item) = target(state, label, named)?;
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
    let (id, item) = target(state, label, named)?;
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

/// `evaluate_bench_track`: every observation read at the stage the track is in,
/// on a worker thread, with nothing moved.
pub(super) fn evaluate_bench_track(
    state: &mut AppState,
    label: &str,
    named: Option<&str>,
    search_px: Option<f64>,
) -> Outcome {
    let (id, item) = match target(state, label, named) {
        Ok(target) => target,
        Err(error) => return Outcome::Done(Err(error)),
    };
    let since = state.action_log.revision();
    match state.start_bench_evaluate(id, &item, search_px) {
        Err(message) => Outcome::Done(Err(ToolError::new(message))),
        Ok(()) => started(state, id, since),
    }
}

/// `fit_bench_track`: the track localized, re-triangulated, re-fused and read
/// back, on a worker thread.
pub(super) fn fit_bench_track(
    state: &mut AppState,
    label: &str,
    named: Option<&str>,
    search_px: Option<f64>,
) -> Outcome {
    let (id, item) = match target(state, label, named) {
        Ok(target) => target,
        Err(error) => return Outcome::Done(Err(error)),
    };
    let since = state.action_log.revision();
    match state.start_bench_fit(id, &item, search_px) {
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
) -> Outcome {
    let (id, item) = match target(state, label, named) {
        Ok(target) => target,
        Err(error) => return Outcome::Done(Err(error)),
    };
    let since = state.action_log.revision();
    match state.start_bench_stage(id, &item, stage) {
        Err(message) => Outcome::Done(Err(ToolError::new(message))),
        Ok(()) => started(state, id, since),
    }
}

/// `search_bench_track_descriptors`: the images that hold the patch around one
/// observation, each added as a candidate, on a worker thread.
pub(super) fn search_bench_track_descriptors(
    state: &mut AppState,
    label: &str,
    named: Option<&str>,
    observation: usize,
    radius_px: Option<f64>,
    min_inliers: Option<usize>,
) -> Outcome {
    let (id, item) = match target(state, label, named) {
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
/// track's patch matches it, each added as a candidate, on a worker thread.
pub(super) fn search_bench_track_geometry(
    state: &mut AppState,
    label: &str,
    named: Option<&str>,
    observation: usize,
) -> Outcome {
    let (id, item) = match target(state, label, named) {
        Ok(target) => target,
        Err(error) => return Outcome::Done(Err(error)),
    };
    let since = state.action_log.revision();
    match state.start_bench_geometry_search(id, &item, observation) {
        Err(message) => Outcome::Done(Err(ToolError::new(message))),
        Ok(()) => started(state, id, since),
    }
}

/// `open_sift_index`: the `.kdf` a search queries, adopted for one node.
///
/// Not an edit and not a version: the index is a file beside the node's `.sfmr`
/// and a handle on it, and nothing about the reconstruction or the bench moves.
/// So the reply is the index itself rather than a version, and there is nothing
/// for `undo` to take back. A file that is not an index of this node opens all
/// the same and reports `state: "stale"` with the sentence saying why.
pub(super) fn open_sift_index(state: &mut AppState, label: &str, path: Option<&str>) -> JsonReply {
    let id = resolve_reconstruction(state, Some(label))?;
    state
        .open_sift_index(id, path.map(std::path::PathBuf::from))
        .map_err(ToolError::new)?;
    Ok(json!({
        "reconstruction_label": node_label(state, id),
        "sift_index": sift_index(state, id),
    }))
}

/// `close_sift_index`: the open forest let go of, the file left where it is.
pub(super) fn close_sift_index(state: &mut AppState, label: &str) -> JsonReply {
    let id = resolve_reconstruction(state, Some(label))?;
    state.close_sift_index(id).map_err(ToolError::new)?;
    Ok(json!({
        "reconstruction_label": node_label(state, id),
        "sift_index": sift_index(state, id),
    }))
}

/// `build_sift_index`: every `.sift` file of the node indexed, written beside
/// its `.sfmr` and opened, on a worker thread.
pub(super) fn build_sift_index(state: &mut AppState, label: &str, path: Option<&str>) -> Outcome {
    let id = match resolve_reconstruction(state, Some(label)) {
        Ok(id) => id,
        Err(error) => return Outcome::Done(Err(error)),
    };
    match state.start_build_sift_index(id, path.map(std::path::PathBuf::from)) {
        Err(message) => Outcome::Done(Err(ToolError::new(message))),
        Ok(()) => started_or(state, id, |state| {
            Ok(json!({
                "reconstruction_label": node_label(state, id),
                "sift_index": sift_index(state, id),
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
        node: id,
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
        .ok_or_else(|| ToolError::new("That reconstruction is no longer loaded."))?;
    let item = match named {
        Some(item) => item.to_string(),
        None => crate::bench::active_track_label(bench)
            .map(str::to_string)
            .ok_or_else(|| {
                ToolError::new(format!(
                    "No track is active on {}'s bench. Name one with track, activate one with \
                     activate_bench_item, or put one on with create_bench_track or \
                     create_bench_cluster.",
                    node_label(state, label_of_node)
                ))
            })?,
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
             measured it and it carries no seed. evaluate_bench_track measures it."
        ))
    })?;
    Ok((
        crate::scene::ImageRef::new(label_of_node, row.image as usize),
        pixel,
    ))
}

// ── Resolution ──────────────────────────────────────────────────────────

/// The node a call named, and the item it acts on: the one it named, or the
/// active track of that node's bench.
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
    let active = state
        .bench(id)
        .and_then(|bench| crate::bench::active_track_label(bench))
        .map(str::to_string)
        .ok_or_else(|| {
            ToolError::new(format!(
                "No track is active on {}'s bench. Name one with track, activate one with \
                 activate_bench_item, or put one on with create_bench_track or \
                 create_bench_cluster.",
                node_label(state, id)
            ))
        })?;
    Ok((id, active))
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

fn thresholds(bars: &Thresholds) -> Value {
    json!({
        "min_zncc": bars.min_zncc,
        "max_shift_px": bars.max_shift_px,
        "max_keypoint_uncertainty": bars.max_keypoint_uncertainty,
        "min_relative_zncc": bars.min_relative_zncc,
    })
}

/// What the stage the track is in carries, beside the observations: the
/// template's cut at the cluster stage, the patch at the track stage.
///
/// The template's samples and the consensus bitmap are reported as present or
/// absent rather than sent: they are pictures, and this surface is not a data
/// channel.
fn stage_data(track: &EditableTrack) -> Value {
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
                "frame_fitted": payload.placement.is_some(),
                "bitmap_fused": payload.bitmap.is_some(),
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
/// keypoint a reading wrote, else the refined cluster position, else the seed
/// the step that proposed it left. It is a top level field rather than
/// something a caller assembles out of the two measurement blocks below,
/// because every reader of this surface wants the one answer those blocks are
/// read for -- a candidate a search has just added has no keypoint at all, and
/// an agent should not have to know which slot to fall back to before it can
/// look at one.
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
        "shift_px": finite(measured.shift_px),
        "localizability": finite(measured.localizability),
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
fn track_measurement(observation: &Observation) -> Value {
    let Some(measured) = observation.track.as_ref() else {
        return Value::Null;
    };
    json!({
        "keypoint": measured.keypoint,
        "zncc": finite(measured.zncc),
        "seed_shift_px": finite(measured.seed_shift_px),
        "projection_offset_px": finite(measured.projection_offset_px),
        "reprojection_error": finite(measured.reprojection_error),
        "ray_angle_deg": finite(measured.ray_angle_deg),
        "localizability": finite(measured.localizability),
        // Present only when the last fit refused the walk and left this sighting
        // at its seed; the number is how far the correlation peak sat.
        "walked_px": finite(measured.walked_px),
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
