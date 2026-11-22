// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The measurement tools: `list_measurements` and `get_measurement`, which
//! read what the Scene tree's Measurements rows and Measurement View show, and
//! `select_measurement`, which selects a measurement, or one of its
//! endpoints, as a click on its row or its block does.
//!
//! Every reading is the node's cached one ([`crate::measurements::readings`]),
//! so the wire, the tree and the panel report the same numbers.

use serde_json::{json, Value};
use sfmtool_core::reconstruction::measurement_endpoint::{
    CandidateStanding, EndpointCandidate, EndpointPosition, EndpointRefusal, EndpointResolution,
    ResolvedDistanceMeasurement,
};

use super::{render, resolve_reconstruction, JsonReply, ToolError};
use crate::measurements::{
    self, DistanceResult, DistanceRow, EndpointSide, MeasurementKind, MeasurementRef,
};
use crate::scene::{ReconId, SceneNode};
use crate::state::AppState;

/// `list_measurements`: every measurement of every kind of one
/// reconstruction, each with the reading its Scene tree row shows.
pub(super) fn list_measurements(state: &AppState, reconstruction_label: Option<&str>) -> JsonReply {
    let id = resolve_reconstruction(state, reconstruction_label)?;
    let node = state.node(id).expect("just resolved");
    let recon = node.recon();
    let unit = recon.metadata.world_space_unit.as_deref();
    let readings = measurements::readings(node);
    let rows = measurements::distance_rows(recon);
    let listed: Vec<Value> = rows
        .iter()
        .enumerate()
        .filter_map(|(index, row)| {
            let resolved = readings.distance.get(index)?;
            Some(summary(
                node,
                MeasurementRef::distance(id, index),
                row,
                resolved,
                unit,
            ))
        })
        .collect();
    Ok(json!({
        "reconstruction_label": node.label,
        "world_space_unit": unit,
        "measurement_counts": counts(recon),
        "measurements": listed,
    }))
}

/// `get_measurement`: one measurement whole. For a distance, both endpoints'
/// images, pixels and resolutions with every candidate, the result block and
/// the header's comparison line.
pub(super) fn get_measurement(
    state: &AppState,
    reconstruction_label: Option<&str>,
    kind: MeasurementKind,
    index: usize,
) -> JsonReply {
    let id = resolve_reconstruction(state, reconstruction_label)?;
    let measurement = resolve_measurement(state, id, kind, index)?;
    let node = state.node(id).expect("just resolved");
    let recon = node.recon();
    let unit = recon.metadata.world_space_unit.as_deref();
    let readings = measurements::readings(node);
    let rows = measurements::distance_rows(recon);
    let row = &rows[index];
    let resolved = &readings.distance[index];
    let mut out = summary(node, measurement, row, resolved, unit);
    let object = out.as_object_mut().expect("a summary is an object");
    let endpoints: Vec<Value> = EndpointSide::BOTH
        .into_iter()
        .map(|side| endpoint_detail(node, row, resolved, side, unit))
        .collect();
    object.insert("endpoints".into(), Value::Array(endpoints));
    // The header's comparison line, null with fewer than two readings.
    let median = measurements::scale_comparison(&rows, &readings.distance, index).map(|line| {
        json!({
            "count": line.count,
            "median_unit_per_scene_unit": line.median_unit_per_scene_unit,
            "deviation_percent": line.deviation_percent,
            "text": measurements::scale_comparison_text(&line, row.unit),
        })
    });
    object.insert("median".into(), median.unwrap_or(Value::Null));
    Ok(out)
}

/// `select_measurement`: select a measurement as a click on its Scene tree row
/// does, or, naming an endpoint, that endpoint as a click on its block in
/// Measurement View does, including the reveal of its pixel in Image Detail.
pub(super) fn select_measurement(
    state: &mut AppState,
    reconstruction_label: Option<&str>,
    kind: MeasurementKind,
    index: usize,
    endpoint: Option<EndpointSide>,
) -> JsonReply {
    let id = resolve_reconstruction(state, reconstruction_label)?;
    let measurement = resolve_measurement(state, id, kind, index)?;
    match endpoint {
        Some(side) => state.select_endpoint(measurement, side),
        None => state.select_measurement(measurement),
    }
    render::selection_reply(state)
}

/// The measurement `kind` `index` names in node `id`, or a refusal that says
/// how many there are.
fn resolve_measurement(
    state: &AppState,
    id: ReconId,
    kind: MeasurementKind,
    index: usize,
) -> Result<MeasurementRef, ToolError> {
    let node = state.node(id).expect("resolved by the caller");
    let count = match kind {
        MeasurementKind::Distance => node
            .recon()
            .image_set
            .measurements
            .distance_measurement_count(),
    };
    if index >= count {
        return Err(ToolError::new(format!(
            "{} stores {count} {} measurement{}, so there is no measurement_index {index}",
            node.label,
            kind.wire_name(),
            if count == 1 { "" } else { "s" }
        )));
    }
    Ok(match kind {
        MeasurementKind::Distance => MeasurementRef::distance(id, index),
    })
}

/// A reconstruction's `measurement_counts` block: how many measurements of
/// each kind it stores.
pub(super) fn counts(recon: &sfmtool_core::SfmrReconstruction) -> Value {
    json!({ "distance": recon.image_set.measurements.distance_measurement_count() })
}

/// One measurement as `list_measurements` lists it: its kind and index, its
/// label, its stated distance, the Scene tree row's text, the result and each
/// endpoint's image and pixel with whether it resolved.
fn summary(
    node: &SceneNode,
    measurement: MeasurementRef,
    row: &DistanceRow,
    resolved: &ResolvedDistanceMeasurement,
    unit: Option<&str>,
) -> Value {
    let recon = node.recon();
    let endpoints: Vec<Value> = EndpointSide::BOTH
        .into_iter()
        .map(|side| {
            let image = row.image(side);
            json!({
                "endpoint": side.wire_name(),
                "camera_image": image,
                "name": recon.image_set.images.get(image as usize).map(|i| i.name.clone()),
                "pixel": row.pixel(side),
                "resolved": measurements::endpoint(resolved, side).is_ok(),
            })
        })
        .collect();
    json!({
        "measurement_kind": measurement.kind.wire_name(),
        "measurement_index": measurement.index(),
        "label": row.label,
        "stated_distance": row.stated,
        "unit": row.unit.name(),
        "row_text": measurements::row_text(measurement, row, resolved, unit),
        "result": result(measurements::distance_result(row, resolved, unit)),
        "endpoints": endpoints,
    })
}

/// The result block as the wire reports it.
fn result(result: DistanceResult) -> Value {
    match result {
        DistanceResult::WithUnit {
            reconstructed,
            error,
            error_percent,
            implied_scale,
        } => json!({
            "state": "with_unit",
            "reconstructed_distance": reconstructed,
            "error": error,
            "error_percent": error_percent,
            "implied_scale": implied_scale,
        }),
        DistanceResult::SceneUnits {
            scene_distance,
            unit_per_scene_unit,
        } => json!({
            "state": "scene_units",
            "scene_distance": scene_distance,
            "unit_per_scene_unit": unit_per_scene_unit,
        }),
        DistanceResult::AtInfinity(side) => json!({
            "state": "at_infinity",
            "endpoint": side.wire_name(),
        }),
        DistanceResult::Refused(side) => json!({
            "state": "refused",
            "endpoint": side.wire_name(),
        }),
    }
}

/// One endpoint as `get_measurement` reports it: its image and pixel, and its
/// resolution with every candidate, or the refusal's sentence.
fn endpoint_detail(
    node: &SceneNode,
    row: &DistanceRow,
    resolved: &ResolvedDistanceMeasurement,
    side: EndpointSide,
    unit: Option<&str>,
) -> Value {
    let recon = node.recon();
    let image = row.image(side);
    let (resolution, refusal) = match measurements::endpoint(resolved, side) {
        Ok(resolution) => (Some(resolution_json(node, resolution)), None),
        Err(refusal) => (None, Some(refusal_json(refusal))),
    };
    json!({
        "endpoint": side.wire_name(),
        "camera_image": image,
        "name": recon.image_set.images.get(image as usize).map(|i| i.name.clone()),
        "pixel": row.pixel(side),
        "position_unit": unit,
        "resolution": resolution,
        "refusal": refusal,
    })
}

/// A resolution: the chosen candidate's index and every candidate.
fn resolution_json(node: &SceneNode, resolution: &EndpointResolution) -> Value {
    json!({
        "chosen": resolution.chosen,
        "candidates": resolution
            .candidates
            .iter()
            .map(|candidate| candidate_json(node, candidate))
            .collect::<Vec<_>>(),
    })
}

/// One candidate: its point, observation, texel, position or direction,
/// depth and standing.
fn candidate_json(node: &SceneNode, candidate: &EndpointCandidate) -> Value {
    let (position, direction) = match candidate.position {
        EndpointPosition::Finite { position } => (Some(position), None),
        EndpointPosition::AtInfinity { direction } => (None, Some(direction)),
    };
    let (standing, exclusion, cost) = match candidate.standing {
        CandidateStanding::Alone => ("alone", None, None),
        CandidateStanding::Ranked(cost) => (
            "ranked",
            None,
            Some(json!({
                "localization": cost.localization,
                "localization_read": cost.localization_read,
                "capped_radius": cost.capped_radius,
                "offset": cost.offset,
                "angle": cost.angle,
                "agreement": cost.agreement,
                "cost": cost.cost,
            })),
        ),
        CandidateStanding::Excluded(why) => ("excluded", Some(why.name()), None),
    };
    json!({
        "point": candidate.point,
        "point_id": crate::scene::point_id(node, candidate.point as usize),
        "observation": candidate.observation,
        "patch_coords": candidate.patch_coords,
        "position": position,
        "direction": direction,
        // `null` for a square at infinity, which JSON has no number for.
        "depth": candidate.depth.is_finite().then_some(candidate.depth),
        "standing": standing,
        "exclusion": exclusion,
        "cost": cost,
    })
}

/// A refusal: its sentence, which names the remedy, and its short form.
fn refusal_json(refusal: &EndpointRefusal) -> Value {
    json!({
        "sentence": refusal.to_string(),
        "short": measurements::refusal_short(refusal),
    })
}
