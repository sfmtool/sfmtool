// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The measurement tools: `list_measurements` and `get_measurement`, which
//! read what the Scene tree's Measurements rows and Measurement View show;
//! `select_measurement`, which selects a measurement, or one of its
//! endpoints, as a click on its row or its block does; and the ones that
//! edit, `put_measurement_on_bench`, `create_bench_measurement`,
//! `set_bench_measurement_label`,
//! `set_bench_measurement_distance`, `set_bench_measurement_endpoint`,
//! `commit_bench_measurement` and `delete_measurement`, each one `AppState`
//! call the panel's gesture makes; `get_measurement_endpoint_views`, the
//! offers of the panel's *Switch image…* popup; and
//! `get_point_reference_pixel`, the pixel the 3D view's point menu sets an
//! endpoint to where its right-click was not on the point's drawn patch.
//!
//! A stored measurement's reading is the node's cached one
//! ([`crate::measurements::readings`]), so the wire, the tree and the panel
//! report the same numbers; an item's is its live reading
//! ([`crate::measurements::item_reading`]), as Measurement View's edited mode
//! reads it.

use serde_json::{json, Value};
use sfmtool_core::reconstruction::measurement_endpoint::{
    endpoint_views, CandidateStanding, EndpointCandidate, EndpointPosition, EndpointRefusal,
    EndpointResolution, MeasurementEndpointOptions, ResolvedDistanceMeasurement,
};

use super::{edit, render, resolve_reconstruction, JsonReply, ToolError};
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
    let rows = measurements::distance_rows(node.edited().measurements());
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
        "measurement_counts": counts(node.edited()),
        "measurements": listed,
    }))
}

/// `get_measurement_scale`: the scale one reconstruction's stored distance
/// measurements give it in `unit`, or in the unit the Scale to Median
/// Measurement dialog opens on, with what the dialog shows of it.
///
/// The factor and the statistics are `null` where the scale is an error, and
/// `refusal` then says why; `excluded` lists every measurement that gives no
/// factor either way. `apply_refusal` is why
/// `scale_reconstruction_to_median_measurement` would be refused right now,
/// in the words the greyed `Apply` carries.
pub(super) fn get_measurement_scale(
    state: &AppState,
    reconstruction_label: Option<&str>,
    unit: Option<sfmtool_core::LengthUnit>,
) -> JsonReply {
    use crate::measurement_scale_prompt::{self as prompt, ScaleView};
    let id = resolve_reconstruction(state, reconstruction_label)?;
    let node = state.node(id).expect("just resolved");
    let unit = unit.unwrap_or_else(|| prompt::default_unit(node));
    let view = ScaleView::of(node, unit);
    let label = |index: usize| view.rows.get(index).map(|row| row.label.clone());
    let excluded =
        |list: &[sfmtool_core::reconstruction::measurement_scale::ExcludedMeasurement]| {
            list.iter()
                .map(|e| {
                    json!({
                        "measurement_kind": MeasurementKind::Distance.name(),
                        "measurement_index": e.index,
                        "label": label(e.index),
                        "reason": prompt::sentence(&e.reason),
                    })
                })
                .collect::<Vec<Value>>()
        };
    let apply_refusal = state
        .edit_refusal(id)
        .or_else(|| prompt::refusal(node))
        .or_else(|| view.scale_refusal());
    let mut reply = json!({
        "reconstruction_label": node.label,
        "world_space_unit": view.file_unit,
        "unit": unit.name(),
        "disagreement_percent":
            sfmtool_core::reconstruction::measurement_scale::SCALE_DISAGREEMENT_PERCENT,
        "apply_refusal": apply_refusal,
    });
    let fields = match &view.scale {
        Ok(scale) => {
            let s = &scale.statistics;
            let file_unit = view
                .file_unit
                .as_deref()
                .and_then(sfmtool_core::LengthUnit::from_name);
            json!({
                "refusal": null,
                "factor": scale.factor(),
                "scale_text": prompt::scale_text(unit, scale.factor()),
                "correction": file_unit.map(|f| prompt::correction(f, unit, scale.factor())),
                "correction_text":
                    prompt::correction_text(view.file_unit.as_deref(), unit, scale.factor()),
                "statistics": {
                    "count": s.count,
                    "total": s.count + scale.excluded.len(),
                    "median": s.median,
                    "mean": s.mean,
                    "sample_std": s.sample_std,
                    "sample_std_percent": s.sample_std_percent,
                    "median_absolute_deviation": s.median_absolute_deviation,
                    "min": s.min,
                    "max": s.max,
                    "max_abs_deviation_percent": s.max_abs_deviation_percent,
                    "disagrees": s.disagrees,
                },
                "factors": scale.factors.iter().map(|f| {
                    let row = view.rows.get(f.index);
                    json!({
                        "measurement_kind": MeasurementKind::Distance.name(),
                        "measurement_index": f.index,
                        "label": label(f.index),
                        "stated_distance": row.map(|r| r.stated),
                        "stated_unit": row.map(|r| r.unit.name()),
                        "stated_in_unit": f.stated,
                        "scene_distance": f.scene_distance,
                        "factor": f.factor,
                        "deviation_percent": f.deviation_percent,
                    })
                }).collect::<Vec<Value>>(),
                "excluded": excluded(&scale.excluded),
            })
        }
        Err(e) => {
            let list = match e {
                sfmtool_core::reconstruction::measurement_scale::DistanceMeasurementScaleError::NoUsableMeasurement { excluded } => excluded.as_slice(),
                _ => &[],
            };
            json!({
                "refusal": prompt::sentence(e),
                "factor": null,
                "scale_text": null,
                "correction": null,
                "correction_text": null,
                "statistics": null,
                "factors": [],
                "excluded": excluded(list),
            })
        }
    };
    let object = reply.as_object_mut().expect("an object");
    for (key, value) in fields.as_object().expect("an object") {
        object.insert(key.clone(), value.clone());
    }
    Ok(reply)
}

/// `get_measurement`: one measurement whole. For a distance, both endpoints'
/// images, pixels and resolutions with every candidate, the result block and
/// the header's comparison line.
///
/// Naming an `item` instead gives a measurement item on the bench with its
/// live readings, as Measurement View's edited mode shows it
/// ([`item_detail`]).
pub(super) fn get_measurement(
    state: &AppState,
    reconstruction_label: Option<&str>,
    kind: MeasurementKind,
    index: Option<usize>,
    item: Option<&str>,
) -> JsonReply {
    let id = resolve_reconstruction(state, reconstruction_label)?;
    let index = match (index, item) {
        (_, Some(item)) => return item_detail(state, id, item),
        (Some(index), None) => index,
        (None, None) => {
            return Err(ToolError::new(
                "Name a stored measurement with measurement_index, or an item on the bench \
                 with item.",
            ))
        }
    };
    let measurement = resolve_measurement(state, id, kind, index)?;
    let node = state.node(id).expect("just resolved");
    let recon = node.recon();
    let unit = recon.metadata.world_space_unit.as_deref();
    let readings = measurements::readings(node);
    let rows = measurements::distance_rows(node.edited().measurements());
    let row = &rows[index];
    let resolved = &readings.distance[index];
    let mut out = summary(node, measurement, row, resolved, unit);
    let object = out.as_object_mut().expect("a summary is an object");
    let endpoints: Vec<Value> = EndpointSide::BOTH
        .into_iter()
        .map(|side| {
            endpoint_detail(
                node,
                side,
                row.image(side),
                row.pixel(side),
                measurements::endpoint(resolved, side),
                unit,
            )
        })
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

/// `get_measurement` naming an item: the measurement item called `item` on
/// node `id`'s bench, with its live readings.
///
/// The fields of a stored measurement's reply where the item has them, plus
/// `item`, `focused`, and `origin`, the row its origin names at the cursor
/// (`null` for an item with no row to replace, whose commit appends one). An
/// endpoint not placed is `null` in `endpoints`, and with one the result is
/// `null`; the median line compares the item with every stored measurement
/// but its origin.
fn item_detail(state: &AppState, id: ReconId, item: &str) -> JsonReply {
    let node = state.node(id).expect("resolved by the caller");
    let bench = node.history.current_bench();
    let measurement = match bench.get(item) {
        Some(sfmtool_core::bench::BenchItem::Measurement(m)) => m,
        Some(sfmtool_core::bench::BenchItem::Track(_)) => {
            return Err(ToolError::new(
                crate::bench::measurement::not_a_measurement(item),
            ))
        }
        None => {
            return Err(ToolError::new(format!(
                "Nothing on {}'s bench is called {item}.",
                node.label
            )))
        }
    };
    let unit = node.recon().metadata.world_space_unit.as_deref();
    let reading = measurements::item_reading(node.edited(), measurement);
    let origin = state.resolved_measurement_origin(node, measurement);
    let endpoints = measurement
        .as_distance()
        .map_or([None, None], |d| d.endpoints);
    let endpoint_values: Vec<Value> = EndpointSide::BOTH
        .into_iter()
        .map(|side| {
            match (
                endpoints[side.index()],
                reading.endpoints[side.index()].as_ref(),
            ) {
                (Some(endpoint), Some(resolution)) => {
                    endpoint_detail(node, side, endpoint.image, endpoint.pixel, resolution, unit)
                }
                _ => Value::Null,
            }
        })
        .collect();
    let row = measurements::item_row(measurement);
    let result_value = match (&row, &reading.resolved) {
        (Some(row), Some(resolved)) => result(measurements::distance_result(row, resolved, unit)),
        _ => Value::Null,
    };
    let median = match (&row, &reading.resolved) {
        (Some(row), Some(resolved)) => {
            let rows = measurements::distance_rows(node.edited().measurements());
            let stored = measurements::readings(node);
            measurements::item_scale_comparison(&rows, &stored.distance, origin, row, resolved).map(
                |line| {
                    json!({
                        "count": line.count,
                        "median_unit_per_scene_unit": line.median_unit_per_scene_unit,
                        "deviation_percent": line.deviation_percent,
                        "text": measurements::scale_comparison_text(&line, row.unit),
                    })
                },
            )
        }
        _ => None,
    };
    let distance = measurement.as_distance().and_then(|d| d.distance);
    Ok(json!({
        "reconstruction_label": node.label,
        "item": item,
        "focused": state.focused_item_label(id) == Some(item),
        "measurement_kind": measurement.kind().name(),
        "origin": origin.map(|index| json!({ "measurement_index": index })),
        "label": measurement.label,
        "stated_distance": distance.map(|(d, _)| d),
        "unit": distance.map(|(_, u)| u.name()),
        "result": result_value,
        "endpoints": endpoint_values,
        "median": median,
    }))
}

/// `put_measurement_on_bench`: the Measurements row's double-click without
/// the raise and the framing ([`AppState::put_measurement_on_bench`]).
pub(super) fn put_measurement_on_bench(
    state: &mut AppState,
    reconstruction_label: &str,
    kind: MeasurementKind,
    index: usize,
) -> JsonReply {
    let id = resolve_reconstruction(state, Some(reconstruction_label))?;
    let measurement = resolve_measurement(state, id, kind, index)?;
    let mut put = None;
    let mut reply = edit::edited(state, id, |state| {
        state
            .put_measurement_on_bench(measurement)
            .map(|done| put = Some(done))
    })?;
    let put = put.expect("a put that succeeded named its item");
    let object = reply.as_object_mut().expect("an edit reply is an object");
    if !put.made {
        // The focus pushes no version, so without this the only sentence in
        // the reply would be the `label` of the version the node stands at,
        // which an earlier step wrote; `create_bench_track` answers the same.
        object.insert(
            "report".into(),
            json!(format!(
                "Put {} on the bench: no effect, it is on the bench already as {}, now the \
                 focused item",
                measurement.lower_title(),
                put.label
            )),
        );
    }
    object.insert("item".into(), json!(put.label));
    Ok(reply)
}

/// `create_bench_measurement`: *New Measurement from Here* without the raise
/// ([`AppState::start_measurement`]): a measurement item on the node's bench
/// with endpoint A at `pixel` of the camera image named, focused. The reply
/// names the item and carries the endpoint's image and pixel.
pub(super) fn create_bench_measurement(
    state: &mut AppState,
    reconstruction_label: &str,
    selector: &super::CameraImageSel,
    pixel: [f64; 2],
) -> JsonReply {
    let id = resolve_reconstruction(state, Some(reconstruction_label))?;
    let image = super::resolve_camera_image(state, id, selector)?;
    let mut label = None;
    let mut reply = edit::edited(state, id, |state| {
        state
            .start_measurement(image, pixel)
            .map(|made| label = Some(made))
    })?;
    let object = reply.as_object_mut().expect("an edit reply is an object");
    object.insert(
        "item".into(),
        json!(label.expect("a start that succeeded named its item")),
    );
    object.insert("camera_image".into(), json!(image.index()));
    object.insert("pixel".into(), json!(pixel));
    Ok(reply)
}

/// `commit_bench_measurement`: Measurement View's *Commit*
/// ([`AppState::commit_bench_measurement`]) on the item named, or on the
/// focused item when none is.
pub(super) fn commit_bench_measurement(
    state: &mut AppState,
    reconstruction_label: &str,
    named: Option<&str>,
) -> JsonReply {
    let (id, item) = measurement_target(state, reconstruction_label, named)?;
    let mut committed = None;
    let mut reply = edit::edited(state, id, |state| {
        state
            .commit_bench_measurement(id, &item)
            .map(|done| committed = Some(done))
    })?;
    let committed = committed.expect("a commit that succeeded named its row");
    let object = reply.as_object_mut().expect("an edit reply is an object");
    object.insert("item".into(), json!(item));
    object.insert(
        "measurement".into(),
        json!({
            "measurement_kind": committed.measurement.kind.name(),
            "measurement_index": committed.measurement.index(),
            "replaced": committed.replaced,
        }),
    );
    Ok(reply)
}

/// The node a measurement item tool names, and the item: the one named, or
/// the focused item where none is.
fn measurement_target(
    state: &AppState,
    reconstruction_label: &str,
    named: Option<&str>,
) -> Result<(ReconId, String), ToolError> {
    let id = resolve_reconstruction(state, Some(reconstruction_label))?;
    let item = match named {
        Some(item) => item.to_string(),
        None => state
            .focused_item_label(id)
            .map(str::to_string)
            .ok_or_else(|| {
                ToolError::new(format!(
                    "No item on {reconstruction_label}'s bench is focused. Name one with item, \
                     focus one with focus_bench_item, or put one on with \
                     put_measurement_on_bench."
                ))
            })?,
    };
    Ok((id, item))
}

/// `set_bench_measurement_label`: the label field of Measurement View's
/// edited mode ([`AppState::set_bench_measurement_label`]), on the item
/// named or the focused one. The reply names the item and carries the
/// measurement's label as it stands after the call.
pub(super) fn set_bench_measurement_label(
    state: &mut AppState,
    reconstruction_label: &str,
    named: Option<&str>,
    label: &str,
) -> JsonReply {
    let (id, item) = measurement_target(state, reconstruction_label, named)?;
    let mut reply = edit::edited(state, id, |state| {
        state
            .set_bench_measurement_label(id, &item, label)
            .map(|_| ())
    })?;
    let object = reply.as_object_mut().expect("an edit reply is an object");
    object.insert("item".into(), json!(item));
    object.insert("measurement_label".into(), json!(label));
    Ok(reply)
}

/// `set_bench_measurement_distance`: the stated distance field of Measurement
/// View's edited mode ([`AppState::set_bench_measurement_distance`]), on the
/// item named or the focused one. The reply names the item and carries the
/// stated distance and its unit as they stand after the call.
pub(super) fn set_bench_measurement_distance(
    state: &mut AppState,
    reconstruction_label: &str,
    named: Option<&str>,
    distance: &str,
) -> JsonReply {
    let (id, item) = measurement_target(state, reconstruction_label, named)?;
    let mut set = None;
    let mut reply = edit::edited(state, id, |state| {
        state
            .set_bench_measurement_distance(id, &item, distance)
            .map(|done| set = Some(done))
    })?;
    let (stated, unit) = set
        .expect("a step that succeeded said what it set")
        .distance;
    let object = reply.as_object_mut().expect("an edit reply is an object");
    object.insert("item".into(), json!(item));
    object.insert("stated_distance".into(), json!(stated));
    object.insert("unit".into(), json!(unit.name()));
    Ok(reply)
}

/// `get_measurement_endpoint_views`: the offers of Measurement View's *Switch
/// image…* popup for endpoint `side` of the item named, or of the focused one
/// (core's `endpoint_views`), in the popup's order: the chosen candidate's
/// point first.
///
/// Refused, in the popup button's words, where the endpoint is not placed or
/// the resolver refuses it. Not refused on a view-only bench or while a task
/// holds the node, since it changes nothing.
pub(super) fn get_measurement_endpoint_views(
    state: &AppState,
    reconstruction_label: &str,
    named: Option<&str>,
    side: EndpointSide,
) -> JsonReply {
    let (id, item) = measurement_target(state, reconstruction_label, named)?;
    let node = state.node(id).expect("just resolved");
    let measurement = match node.history.current_bench().get(&item) {
        Some(sfmtool_core::bench::BenchItem::Measurement(m)) => m,
        Some(sfmtool_core::bench::BenchItem::Track(_)) => {
            return Err(ToolError::new(
                crate::bench::measurement::not_a_measurement(&item),
            ))
        }
        None => {
            return Err(ToolError::new(format!(
                "Nothing on {}'s bench is called {item}.",
                node.label
            )))
        }
    };
    let refused = |resolution: Option<&Result<EndpointResolution, EndpointRefusal>>| {
        ToolError::new(
            crate::measurement_view::switch_refusal(None, side, resolution)
                .unwrap_or_else(|| "The endpoint has no views.".to_string()),
        )
    };
    let Some(endpoint) = measurement
        .as_distance()
        .and_then(|d| d.endpoints[side.index()])
    else {
        return Err(refused(None));
    };
    let options = MeasurementEndpointOptions::default();
    let views = endpoint_views(node.edited(), endpoint.image, endpoint.pixel, &options)
        .map_err(|refusal| refused(Some(&Err(refusal))))?;
    let recon = node.recon();
    let image_name = |image: u32| {
        recon
            .image_set
            .images
            .get(image as usize)
            .map(|i| i.name.clone())
    };
    let listed: Vec<Value> = views
        .iter()
        .map(|view| {
            json!({
                "point": view.point,
                "point_id": crate::scene::point_id(node, view.point as usize),
                "observation": view.observation,
                "camera_image": view.image,
                "camera_image_name": image_name(view.image),
                "pixel": view.pixel,
                "patch_coords": view.patch_coords,
                "localization": view.localization,
                "viewing_angle_deg": view.viewing_angle_deg,
                "resolves_to": view.resolves_to,
                "resolves_to_point_id": view
                    .resolves_to
                    .map(|point| crate::scene::point_id(node, point as usize)),
                "resolves_elsewhere": view.resolves_elsewhere(),
            })
        })
        .collect();
    Ok(json!({
        "reconstruction_label": node.label,
        "item": item,
        "endpoint": side.wire_name(),
        "camera_image": endpoint.image,
        "camera_image_name": image_name(endpoint.image),
        "pixel": endpoint.pixel,
        "localization_unit": recon.metadata.world_space_unit,
        "views": listed,
    }))
}

/// `get_point_reference_pixel`: point `point`'s reference pixel, the image
/// and keypoint of its reference observation
/// ([`crate::measurements::reference_pixel`]), which *Set Endpoint A Here*
/// and *Set Endpoint B Here* in the 3D view's point menu set an endpoint to
/// where the right-click that opened it was not on the point's drawn patch.
/// The tool has no pointer, so it gives this fallback and never a spot on the
/// patch ([`crate::measurements::spot_pixel`]).
///
/// Refused in the menu entries' words where the point has no patch frame, no
/// reference observation, or is at infinity. It changes nothing, so it is not
/// refused on a view-only bench or while a task holds the node.
pub(super) fn get_point_reference_pixel(
    state: &AppState,
    reconstruction_label: &str,
    point: &crate::goto_point::PointQuery,
) -> JsonReply {
    let id = resolve_reconstruction(state, Some(reconstruction_label))?;
    let point = super::resolve_point_in(state, id, point)?;
    let node = state.node(id).expect("just resolved");
    let reference =
        measurements::reference_pixel(node.edited(), point.point).map_err(ToolError::new)?;
    Ok(json!({
        "reconstruction_label": node.label,
        "point": point.point,
        "point_id": crate::scene::point_id(node, point.index()),
        "observation": reference.observation,
        "camera_image": reference.image,
        "camera_image_name": measurements::image_name(node.recon(), reference.image),
        "pixel": reference.pixel,
    }))
}

/// `set_bench_measurement_endpoint`: move endpoint `side` of the item named,
/// or of the focused one, to `pixel` of the camera image named
/// ([`AppState::set_bench_measurement_endpoint`]). Any pixel inside the image
/// is taken, not only one the popup offers. The reply names the item and
/// carries the endpoint's image and pixel as they stand after the call.
pub(super) fn set_bench_measurement_endpoint(
    state: &mut AppState,
    reconstruction_label: &str,
    named: Option<&str>,
    side: EndpointSide,
    selector: &super::CameraImageSel,
    pixel: [f64; 2],
) -> JsonReply {
    let (id, item) = measurement_target(state, reconstruction_label, named)?;
    let image = super::resolve_camera_image(state, id, selector)?.index() as u32;
    let mut reply = edit::edited(state, id, |state| {
        state
            .set_bench_measurement_endpoint(
                id,
                &item,
                side,
                image,
                pixel,
                crate::bench::measurement::EndpointMove::Set,
            )
            .map(|_| ())
    })?;
    let object = reply.as_object_mut().expect("an edit reply is an object");
    object.insert("item".into(), json!(item));
    object.insert("endpoint".into(), json!(side.wire_name()));
    object.insert("camera_image".into(), json!(image));
    object.insert("pixel".into(), json!(pixel));
    Ok(reply)
}

/// `delete_measurement`: *Delete* in a Measurements row's menu
/// ([`AppState::delete_measurement`]).
pub(super) fn delete_measurement(
    state: &mut AppState,
    reconstruction_label: &str,
    kind: MeasurementKind,
    index: usize,
) -> JsonReply {
    let id = resolve_reconstruction(state, Some(reconstruction_label))?;
    let measurement = resolve_measurement(state, id, kind, index)?;
    edit::edited(state, id, |state| state.delete_measurement(measurement))
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
    let count = kind.count(node.edited().measurements());
    if index >= count {
        return Err(ToolError::new(format!(
            "{} stores {count} {} measurement{}, so there is no measurement_index {index}",
            node.label,
            kind.name(),
            if count == 1 { "" } else { "s" }
        )));
    }
    Ok(match kind {
        MeasurementKind::Distance => MeasurementRef::distance(id, index),
    })
}

/// A reconstruction's `measurement_counts` block: how many measurements of
/// each kind it stores.
pub(super) fn counts(edited: &sfmtool_core::EditedReconstruction) -> Value {
    json!({ "distance": edited.measurements().distance_measurement_count() })
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
                "camera_image_name": recon.image_set.images.get(image as usize).map(|i| i.name.clone()),
                "pixel": row.pixel(side),
                "resolved": measurements::endpoint(resolved, side).is_ok(),
            })
        })
        .collect();
    json!({
        "measurement_kind": measurement.kind.name(),
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
    side: EndpointSide,
    image: u32,
    pixel: [f64; 2],
    resolution: &Result<EndpointResolution, EndpointRefusal>,
    unit: Option<&str>,
) -> Value {
    let recon = node.recon();
    let (resolution, refusal) = match resolution {
        Ok(resolution) => (Some(resolution_json(node, resolution)), None),
        Err(refusal) => (None, Some(refusal_json(refusal))),
    };
    json!({
        "endpoint": side.wire_name(),
        "camera_image": image,
        "camera_image_name": recon.image_set.images.get(image as usize).map(|i| i.name.clone()),
        "pixel": pixel,
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
